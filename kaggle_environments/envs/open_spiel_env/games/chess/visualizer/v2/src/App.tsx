import Meter from './components/Meter';
import StyledChessboard from './components/StyledChessboard';
import Controls from './components/Controls';
import Legend from './components/Legend';
import './App.css';

function App() {
  return (
    <div className="container">
      <Meter />
      <StyledChessboard />
      <Controls />
      <Legend />
    </div>
  );
}

export default App;
