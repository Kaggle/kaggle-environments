import useGoStore from '../stores/useGoStore';

export default function StyledGoboard() {
  const go = useGoStore((state) => state.go);

  console.log(go.score());
  console.log(go.isOver());

  return <div id="board"></div>;
}
