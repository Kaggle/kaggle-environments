import useGameStore from '../stores/useGameStore';

export default function HiddenHeader() {
  const options = useGameStore((state) => state.options);

  const blackName = options?.replay.info?.TeamNames[0] ?? 'Black';
  const whiteName = options?.replay.info?.TeamNames[1] ?? 'White';

  return (
    <h1 className="visually-hidden">
      {blackName} vs. {whiteName}
    </h1>
  );
}
