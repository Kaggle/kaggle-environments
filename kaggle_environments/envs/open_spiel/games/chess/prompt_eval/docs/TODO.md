# Chess Prompt Evaluation TODO

## Current Sprint - Core Runner Implementation
- [ ] **Batch Processing System**: Parallel evaluation runner with JSONL output
- [ ] **Results Storage Format**: Implement the simplified schema we designed
- [ ] **Multiple Prompt Strategies**: Expand beyond basic FEN prompts
- [ ] **Evaluation Pipeline**: End-to-end automation from positions → results

## Model Configuration
- [ ] **Figure out correct temps for each model**: Research and test optimal temperature settings per provider
- [ ] **Token counting accuracy**: Ensure completion_tokens includes thinking tokens where applicable

## Infrastructure & Testing
- [ ] **Integration tests**: End-to-end runner testing with small position sets
- [ ] **Error handling**: Robust API failure recovery and graceful degradation
- [ ] **Progress monitoring**: Real-time stats and ETA during long runs
- [ ] **Resume capability**: Handle interrupted runs gracefully

## Prompt Strategy Development
- [ ] **JSON board representation**: Alternative to FEN-based prompts
- [ ] **Different instruction formats**: Vary the way we ask for moves
- [ ] **Strategy comparison framework**: Easy A/B testing of prompt approaches

## Analysis & Reporting (Future)
- [ ] **Statistical analysis tools**: Compare performance across strategies/models
- [ ] **Position categorization**: Opening/middlegame/endgame performance breakdowns
- [ ] **Visualization**: Charts and reports for experiment results
- [ ] **Cost tracking**: Monitor API usage and expenses across experiments

## Current Blockers
- None identified

## Notes
- One run = one model + one strategy (simplified from original multi-strategy design)
- Focus on legal move rate and move quality (WDL expectation changes) as primary metrics
- Prioritize Gemini models for cost-effectiveness with Google credits