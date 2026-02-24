import useGoStore from '../stores/useGoStore';

export default function StyledGoboard() {
  const go = useGoStore((state) => state.go);

  console.log(go.currentState());

  return <div id="board"></div>;
}
