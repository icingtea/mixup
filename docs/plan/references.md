# References

Consolidated bibliography for the plan. Grouped by topic; the bracketed keys are used
throughout `docs/plan/`.

## Membership & failure detection

- [SWIM] A. Das, I. Gupta, A. Motivala. *SWIM: Scalable Weakly-consistent Infection-style
  Process Group Membership Protocol*. DSN 2002.
  <https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SWIM.pdf>
- [Serf-gossip] HashiCorp. *Serf internals: gossip protocol.*
  <https://www.serf.io/docs/internals/gossip.html>
- [memberlist] HashiCorp. *memberlist — Golang membership library (SWIM implementation).*
  <https://github.com/hashicorp/memberlist>
- [Chandra-Toueg] T. Chandra, S. Toueg. *Unreliable Failure Detectors for Reliable
  Distributed Systems*. JACM 1996. <https://doi.org/10.1145/226643.226647>
- [PhiAccrual] N. Hayashibara, X. Défago, R. Yared, T. Katayama. *The φ Accrual Failure
  Detector*. SRDS 2004. <http://www.jaist.ac.jp/~defago/files/pdf/IS_RR_2004_010.pdf>

## Gossip & epidemic algorithms

- [Demers] A. Demers et al. *Epidemic Algorithms for Replicated Database Maintenance.*
  PODC 1987. <https://doi.org/10.1145/41840.41841>
- [Birman] K. Birman. *The Promise, and Limitations, of Gossip Protocols.* ACM SIGOPS
  Operating Systems Review, 2007. <https://doi.org/10.1145/1317379.1317382>
- [CRDT] M. Shapiro et al. *Conflict-Free Replicated Data Types.* (survey)
  <https://crdt.tech/>

## Leader election & consensus

- [Garcia-Molina] H. Garcia-Molina. *Elections in a Distributed Computing System.* IEEE
  Transactions on Computers, 1982. <https://doi.org/10.1109/TC.1982.1675945>
- [Raft] D. Ongaro, J. Ousterhout. *In Search of an Understandable Consensus Algorithm.*
  USENIX ATC 2014. <https://raft.github.io/raft.pdf> — §5.4.1 (election restriction) is
  the model for the "freshest ledger wins" rule.
- [Lamport] L. Lamport. *Time, Clocks, and the Ordering of Events in a Distributed
  System.* CACM 1978. <https://lamport.azurewebsites.net/pubs/time-clocks.pdf>
- [Brewer-CAP] E. Brewer. *CAP Twelve Years Later: How the "Rules" Have Changed.* InfoQ
  2012. <https://www.infoq.com/articles/cap-twelve-years-later-how-the-rules-have-changed/>
- [PACELC] D. Abadi. *Consistency Tradeoffs in Modern Distributed Database System Design.*
  IEEE Computer 2012. <https://cs-www.cs.yale.edu/homes/dna/papers/abadi-pacelc.pdf>

## MoE models & sparse routing

- [Shazeer-MoE] N. Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer.* ICLR 2017. <https://arxiv.org/abs/1701.06538>
- [GShard] D. Lepikhin et al. *GShard: Scaling Giant Models with Conditional Computation
  and Automatic Sharding.* ICLR 2021. <https://arxiv.org/abs/2006.16668>
- [Switch] W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity.* JMLR 2022.
  <https://arxiv.org/abs/2101.03961>
- [Mixtral] A. Jiang et al. *Mixtral of Experts.* 2024. <https://arxiv.org/abs/2401.04088>
- [Granite] IBM. *Granite 3.1 1B A400M Instruct.*
  <https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-instruct>

## Expert parallelism & dynamic placement

- [DeepSpeed-MoE] S. Rajbhandari et al. *DeepSpeed-MoE: Advancing Mixture-of-Experts
  Inference and Training to Power Next-Generation AI Scale.* ICML 2022.
  <https://arxiv.org/abs/2201.05596>
- [Tutel] C. Hwang et al. *Tutel: Adaptive Mixture-of-Experts at Scale.* MLSys 2023.
  <https://arxiv.org/abs/2206.03382>
- [FasterMoE] J. He et al. *FasterMoE: Modeling and Optimizing Training of Large-Scale
  Dynamic Pre-Trained Models.* PPoPP 2022. <https://arxiv.org/abs/2110.03742>

## Transformers internals & inference

- [HF-generation] Hugging Face. *Generation strategies.*
  <https://huggingface.co/docs/transformers/main/en/generation_strategies>
- [HF-llm-optimization] Hugging Face. *LLM inference optimization (KV cache, etc.).*
  <https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization>
- [safetensors] Hugging Face. *safetensors.*
  <https://github.com/huggingface/safetensors>
- [SpecDec] Y. Leviathan, M. Kalman, Y. Matias. *Fast Inference from Transformers via
  Speculative Decoding.* ICML 2023. <https://arxiv.org/abs/2211.17192>

## Transport (HTTP vs gRPC)

- [gRPC] gRPC Authors. *Introduction to gRPC.*
  <https://grpc.io/docs/what-is-grpc/introduction/>
- [HTTP2] *HTTP/2 (RFC 9113).* <https://www.rfc-editor.org/rfc/rfc9113>
- [httpx] *HTTPX — HTTP client for Python (incl. HTTP/2).*
  <https://www.python-httpx.org/http2/>

## Tooling & ops

- [uv] Astral. *uv documentation.* <https://docs.astral.sh/uv/>
- [uv2nix] pyproject-nix. *uv2nix.* <https://github.com/pyproject-nix/uv2nix>
- [nvidia-toolkit] NVIDIA. *NVIDIA Container Toolkit — install guide.*
  <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- [docker-gpu] Docker. *Runtime options / GPU support.*
  <https://docs.docker.com/engine/containers/resource_constraints/>
- [ChaosEng] *Principles of Chaos Engineering.* <https://principlesofchaos.org/>
