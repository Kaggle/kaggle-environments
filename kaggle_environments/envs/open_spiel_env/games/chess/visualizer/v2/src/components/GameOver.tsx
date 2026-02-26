import useChessStore from '../stores/useChessStore';

export default function GameOver() {
  const chess = useChessStore((state) => state.chess);
  const options = useChessStore((state) => state.options);

  if (!chess.isGameOver()) return;

  const tokens: {
    [fieldName: string]: {
      promptTokens: number;
      generationTokens: number;
      reasoningTokens: number;
      totalTokens: number;
      inputCost: number;
      outputCost: number;
      totalCost: number;
    };
  } = {};

  options?.replay.steps.forEach((step) => {
    const player = step.players.find((player) => player.generateReturns);

    if (!player) return;

    const name = player!.name;

    if (!tokens[name]) {
      tokens[name] = {
        promptTokens: 0,
        generationTokens: 0,
        reasoningTokens: 0,
        totalTokens: 0,
        inputCost: 0,
        outputCost: 0,
        totalCost: 0,
      };
    }

    const ret = JSON.parse(player!.generateReturns![0]);

    const modelName: string = ret.model ?? ret.request_for_logging.model;
    const promptTokens: number = ret.prompt_tokens ?? 0;
    let generationTokens: number = ret.generation_tokens ?? 0;
    const reasoningTokens: number = ret.reasoning_tokens ?? 0;
    let totalTokens: number = ret.total_tokens ?? 0;

    if (modelName.includes('grok') || modelName.includes('gemini')) {
      generationTokens = totalTokens - promptTokens;
    }

    if (totalTokens === 0) {
      totalTokens = promptTokens + generationTokens + reasoningTokens;
    }

    tokens[name].promptTokens += promptTokens;
    tokens[name].generationTokens += generationTokens;
    tokens[name].reasoningTokens += reasoningTokens;
    tokens[name].totalTokens += totalTokens;
  });

  console.log(tokens);

  return null;
}
