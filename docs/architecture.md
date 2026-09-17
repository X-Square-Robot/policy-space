# Runtime Architecture

Policy Space defines the runtime boundary between an evaluation client and a policy service. The current reference integration uses ManaEnv as the simulation client. The client owns environment interaction, robot execution, task configuration, safety, and evaluation. The policy service owns model-specific preprocessing, inference, and optional history or state. Policy Space connects them through versioned observation, action, embodiment, and episode-lifecycle contracts.

The repository homepage uses the publication-quality vector figure in [`docs/assets/policy-space-runtime-overview.svg`](assets/policy-space-runtime-overview.svg). The Mermaid source below provides a lightweight, editable view of the same runtime structure.

```mermaid
flowchart LR
    subgraph C["Simulation Client"]
        E["Environment & Robot"] --> O["Observation"]
        E -. privileged state .-> V["Evaluator"]
    end

    subgraph P["Policy Space"]
        OM["Observation Mapping"]
        PC["Policy-Embodiment Contract"]
        AA["Action Adaptation"]
        EL["Episode Lifecycle"]
        PC -.-> OM
        PC -.-> AA
        EL -.-> OM
        EL -.-> AA
    end

    subgraph S["Policy Service"]
        MA["Model Adapter"] --> PI["Policy Inference"]
        HS["Optional History / State"] <--> PI
    end

    O --> OM --> MA
    PI --> AA --> E

    classDef client fill:#FDF2F8,stroke:#DC8BB5,color:#222;
    classDef core fill:#F3EFFF,stroke:#8D78CF,color:#222;
    classDef service fill:#EEF3FF,stroke:#7C92D9,color:#222;
    class E,O,V client;
    class OM,PC,AA,EL core;
    class MA,PI,HS service;
```

## Runtime responsibilities

- **Simulation client:** produces observations, executes actions, advances the environment, and evaluates task progress.
- **Policy Space:** maps observations, adapts actions, negotiates policy-embodiment contracts, and coordinates the episode lifecycle.
- **Policy service:** converts mapped observations into model-native inputs, runs inference, and maintains optional model history or state.

Privileged simulator state is available only to the evaluator and is not sent to the policy service.

The same protocol can be implemented by robot-control clients for future real-robot deployment.

[简体中文](architecture.zh-CN.md)
