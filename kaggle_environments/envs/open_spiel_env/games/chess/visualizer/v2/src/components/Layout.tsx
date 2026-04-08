import { memo } from 'react';
import GameBoard from './GameBoard';

export default memo(function Layout() {
  return (
    <div id="renderer">
      <GameBoard />
    </div>
  );
});
