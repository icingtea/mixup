"""mixup-tui: an interactive chat + observability terminal UI for the mixup cluster.

Chat like a prompt, see every node's live status in a sidebar (with kill/restart toggles
and the leader badge), and browse tabs for resource time series, expert activation,
predictor accuracy, a project summary, and a single-node (no-distribution) benchmark.
"""

import asyncio
import subprocess
from pathlib import Path

import httpx
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
)

from mixup_core.hosts import node_urls
from mixup_core.startup_config import load_startup_config

POLL = 1.0
SPARK = 120  # samples kept per sparkline


class Sidebar(Static):
    """One node's status card (status, role, experts, kill/restart buttons)."""

    def __init__(self, name: str, url: str) -> None:
        super().__init__(id=f"node-{name}")
        self.name = name
        self.url = url
        self.selected = False


class MixupApp(App):
    """The mixup TUI."""

    TITLE = "mixup"
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("i", "focus_chat", "chat input"),
    ]

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config = load_startup_config(config_path)
        self.urls = node_urls(self.config)
        self.client = httpx.AsyncClient(timeout=3.0)
        self.leader: str | None = None
        self.nodes: dict[str, dict] = {}  # name -> latest /metrics
        self.history: dict[str, list[float]] = {}  # name -> cpu util ring buffer
        self.predictor: dict = {}
        self.experts: dict = {}

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield self._sidebar()
            with TabbedContent(initial="chat"):
                with TabPane("Chat", id="chat"):
                    yield self._chat()
                with TabPane("Nodes", id="nodes"):
                    yield self._nodes_tab()
                with TabPane("Metrics", id="metrics"):
                    yield self._metrics_tab()
                with TabPane("Experts", id="experts"):
                    yield self._experts_tab()
                with TabPane("Predictor", id="predictor"):
                    yield self._predictor_tab()
                with TabPane("Summary", id="summary"):
                    yield self._summary_tab()
                with TabPane("Local", id="local"):
                    yield self._local_tab()
        yield Footer()

    def _sidebar(self) -> VerticalScroll:
        return VerticalScroll(id="sidebar")

    def _chat(self) -> Vertical:
        return Vertical(
            RichLog(id="chat-log", markup=True, wrap=True),
            Input(placeholder="Send a prompt…", id="chat-input"),
            id="chat",
        )

    def _nodes_tab(self) -> VerticalScroll:
        return VerticalScroll(DataTable(id="nodes-table"), id="nodes-tab")

    def _metrics_tab(self) -> VerticalScroll:
        return VerticalScroll(id="metrics-tab")

    def _experts_tab(self) -> VerticalScroll:
        return VerticalScroll(DataTable(id="experts-table"), id="experts-tab")

    def _predictor_tab(self) -> VerticalScroll:
        return VerticalScroll(Static(id="predictor-text"), id="predictor-tab")

    def _summary_tab(self) -> VerticalScroll:
        return VerticalScroll(Static(id="summary-text"), id="summary-tab")

    def _local_tab(self) -> Vertical:
        return Vertical(
            Button("Run local benchmark (no distribution)", id="local-run"),
            Static(id="local-text", markup=True),
            id="local",
        )

    # -- lifecycle --------------------------------------------------------

    async def on_mount(self) -> None:
        self._build_sidebar()
        self._build_sparks()
        self.run_worker(self._poll_loop(), exclusive=False)

    def _build_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar", VerticalScroll)
        for name, url in self.urls.items():
            sidebar.mount(Sidebar(name, url))
            sidebar.mount(Button("kill", id=f"kill-{name}", classes="kill"))
            sidebar.mount(Button("start", id=f"start-{name}", classes="start"))

    def _build_sparks(self) -> None:
        tab = self.query_one("#metrics-tab", VerticalScroll)
        self._sparks: dict[str, Sparkline] = {}
        for name in self.urls:
            tab.mount(Label(f"[b]{name}[/b] cpu"))
            spark = Sparkline([], id=f"spark-{name}")
            tab.mount(spark)
            self._sparks[name] = spark

    # -- polling ----------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            await self._poll_once()
            await asyncio.sleep(POLL)

    async def _poll_once(self) -> None:
        # Leader: highest epoch across /leader projections.
        projections = []
        for url in self.urls.values():
            try:
                projections.append((await self.client.get(f"{url}/leader")).json())
            except httpx.HTTPError:
                continue
        best, best_epoch = None, -1
        for p in projections:
            if p.get("leader") and p.get("epoch", 0) > best_epoch:
                best, best_epoch = p["leader"], p.get("epoch", 0)
        self.leader = best

        # Per-node metrics.
        for name, url in self.urls.items():
            try:
                self.nodes[name] = (await self.client.get(f"{url}/metrics")).json()
                cpu = self.nodes[name]["resources"]["cpu_util"]
                self.history.setdefault(name, []).append(cpu)
                if len(self.history[name]) > SPARK:
                    self.history[name] = self.history[name][-SPARK:]
            except httpx.HTTPError:
                pass

        # Leader-only: predictor + experts.
        if self.leader and self.leader in self.urls:
            lurl = self.urls[self.leader]
            try:
                self.predictor = (await self.client.get(f"{lurl}/predictor")).json()
                self.experts = (await self.client.get(f"{lurl}/experts")).json()
            except httpx.HTTPError:
                pass

        self._refresh()

    def _refresh(self) -> None:
        self._refresh_sidebar()
        self._refresh_nodes()
        self._refresh_metrics()
        self._refresh_experts()
        self._refresh_predictor()
        self._refresh_summary()

    # -- refresh helpers --------------------------------------------------

    def _refresh_sidebar(self) -> None:
        for name in self.urls:
            info = self.nodes.get(name, {})
            role = info.get("role", "?")
            is_leader = name == self.leader
            card = self.query_one(f"#node-{name}", Static)
            badge = "★LEADER " if is_leader else ""
            owned = len(info.get("owned_experts", []))
            cpu = info.get("resources", {}).get("cpu_util", 0.0)
            card.update(f"[b]{badge}{name}[/b]  {role}  {owned} experts  cpu {cpu:.2f}")

    def _refresh_nodes(self) -> None:
        table = self.query_one("#nodes-table", DataTable)
        if not table.columns:
            table.add_columns("node", "role", "leader", "exec", "experts", "cpu", "gpu_free")
        rows = []
        for name in self.urls:
            info = self.nodes.get(name, {})
            r = info.get("resources", {})
            rows.append([
                name,
                info.get("role", "?"),
                "★" if name == self.leader else "",
                info.get("execution", "?"),
                str(len(info.get("owned_experts", []))),
                f"{r.get('cpu_util', 0):.2f}",
                f"{r.get('gpu_free_bytes') or ''}",
            ])
        table.clear()
        table.add_rows(rows)

    def _refresh_metrics(self) -> None:
        for name in self.urls:
            self._sparks[name].data = self.history.get(name, [])

    def _refresh_experts(self) -> None:
        table = self.query_one("#experts-table", DataTable)
        if not table.columns:
            table.add_columns("expert", "activations", "bar")
        counts = self.experts.get("counts", [])
        if not counts:
            return
        ranked = sorted(range(len(counts)), key=lambda e: -counts[e])
        max_c = max(counts) if counts else 1
        rows = []
        for e in ranked[:12]:
            rows.append([str(e), str(int(counts[e])), "█" * int(40 * counts[e] / max_c)])
        table.clear()
        table.add_rows(rows)

    def _refresh_predictor(self) -> None:
        p = self.predictor
        if not p:
            return
        s = p.get("samples", 0)
        t1 = p.get("top1_hits", 0)
        t3 = p.get("top3_hits", 0)
        text = (
            f"model: {p.get('kind', '?')}  version {p.get('version', 0)}\n\n"
            f"samples: {s}\n"
            f"top-1: {t1}/{s} = {t1 / s:.3f}  (misses {s - t1})\n"
            f"top-3: {t3}/{s} = {t3 / s:.3f}  (misses {s - t3})\n"
        )
        self.query_one("#predictor-text", Static).update(text)

    def _refresh_summary(self) -> None:
        leader = self.leader or "none"
        text = (
            f"[b]cluster[/b]\n"
            f"  leader: {leader}\n"
            f"  nodes: {len(self.urls)}\n\n"
            f"[b]predictor[/b]\n"
            f"  kind={self.predictor.get('kind', '?')} "
            f"samples={self.predictor.get('samples', 0)}\n\n"
            f"[b]phases[/b]\n"
            f"  0-6 done; 7 (deploy) deferred\n"
        )
        self.query_one("#summary-text", Static).update(text)

    # -- actions ----------------------------------------------------------

    async def action_focus_chat(self) -> None:
        self.query_one("#chat-input", Input).focus()

    @staticmethod
    def _docker(action: str, name: str) -> str:
        try:
            return subprocess.run(
                ["docker", action, name], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""

    def _handle_node_button(self, action: str, name: str) -> None:
        self._docker(action, name)
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[i]{action} {name}[/i]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("kill-"):
            self._handle_node_button("stop", event.button.id[5:])
        elif event.button.id and event.button.id.startswith("start-"):
            self._handle_node_button("start", event.button.id[6:])
        elif event.button.id == "local-run":
            self.run_worker(self._run_local_bench(), exclusive=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        event.input.value = ""
        if not prompt:
            return
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[b]you>[/b] {prompt}")
        self.run_worker(self._send_generate(prompt), exclusive=False)

    async def _send_generate(self, prompt: str) -> None:
        from mixup_core.types import GenerateRequest

        log = self.query_one("#chat-log", RichLog)
        if not self.leader or self.leader not in self.urls:
            log.write("[red]no leader[/red]")
            return
        url = self.urls[self.leader]
        req = GenerateRequest(prompt=prompt, max_new_tokens=80)
        try:
            resp = await self.client.post(f"{url}/generate", json=req.model_dump(), timeout=None)
            resp.raise_for_status()
            data = resp.json()
            log.write(f"[green]assistant>[/green] {data['text']}")
            log.write(
                f"[i]{data['tokens']} tokens in {data['elapsed_seconds']:.2f}s "
                f"({data['tokens'] / max(data['elapsed_seconds'], 1e-9):.2f} tok/s)[/i]"
            )
        except httpx.HTTPError as exc:
            log.write(f"[red]error: {exc}[/red]")

    async def _run_local_bench(self) -> None:
        from mixup_core.constants import MODEL_DIR
        from mixup_repl.repl import LocalModel

        text = self.query_one("#local-text", Static)
        text.update("[i]loading local model…[/i]")
        model = LocalModel(Path(MODEL_DIR))
        result = await asyncio.to_thread(
            model.generate, "Explain mixture of experts models.", 60
        )
        text.update(
            f"[b]local (no distribution)[/b]\n"
            f"{result[1]} tokens in {result[2]:.2f}s = "
            f"{result[1] / max(result[2], 1e-9):.2f} tok/s\n\n"
            f"{result[0][:400]}"
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Interactive mixup TUI.")
    parser.add_argument("config", nargs="?", default="lab_config.json")
    args = parser.parse_args()
    MixupApp(args.config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
