# widescreen-rcp-prototype
WideScreen: Autonomous Scientific Discovery via Recursive Consensus Protocols (Template-based prototype)

Author: William Alubokho Ashioya  
Status: Template-based prototype (LLM integration planned)  
License: MIT

## Overview

WideScreen demonstrates a multi-agent architecture for autonomous scientific discovery using self-evolving protocol genomes (Recursive Consensus Protocol - RCP).

⚠️ Current Status: This is a template-based demonstration prototype. Hypothesis generation, novelty checking, and experiment design currently use rule-based templates to demonstrate the architecture and workflow orchestration. LLM integration (GPT-4/Claude APIs) is planned for production deployment.

## Real Working Results

Despite template-based ideation, the system executes real numerical experiments:

- Test Problem: Lorenz attractor (stiff ODE system)
- Result: `error: 0.007906` (L² norm vs reference solution)
- Steps: 99,987 function evaluations
- Success Rate: 100% (deterministic, reproducible)

## Architecture

- Ideator Agent: Generates hypotheses (template → LLM-extensible)
- Critic Agent: Evaluates novelty (random score → semantic search)
- Experimenter Agent: Designs numerical experiments
- Interpreter Agent: Analyzes results
- Archivist Agent: Generates LaTeX reports

RCP Genome Evolution: Verification thresholds and interaction rules evolve across iterations based on fitness scores.

## Installation
```bash
git clone https://github.com/Ashioya-ui/widescreen-rcp-prototype
cd widescreen-rcp-prototype
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python run_widescreen.py --domain "stiff ODE solvers" --problem-type ode --max-iterations 5
```

Outputs:
- `outputs/report_iter_X.tex` - LaTeX scientific reports
- `data/iteration_log.json` - Full decision history
- `data/evolution_history.json` - Genome evolution tracking

## Roadmap (Post-Funding)

1. LLM Integration: Replace template modules with GPT-4/Claude
2. Semantic Novelty: Embed hypotheses, compare vs arXiv/PubMed
3. Multi-Domain: Expand to quantum chemistry, materials science
4. Validation: Blind expert evaluation study

## Citation

If you use this work, please cite:
```
Ashioya, W. A. (2025). WideScreen: Autonomous Scientific Discovery via 
Recursive Consensus Protocols. GitHub repository.
```

## Contact

- Email: waa6673@nyu.edu


## License

MIT License - See LICENSE file
```
