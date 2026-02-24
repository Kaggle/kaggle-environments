import { useEffect } from 'react';
import useGoStore from '../stores/useGoStore';

export default function StyledGoboard() {
  const go = useGoStore((state) => state.go);

  useEffect(() => {
    console.log(go.score());
    console.log(go.isOver());
  }, [go]);

  return <div id="board"></div>;
}
