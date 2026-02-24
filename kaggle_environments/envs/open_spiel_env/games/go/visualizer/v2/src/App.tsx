import Controls from './components/Controls';
import StyledGoboard from './components/StyledGoboard';
import './App.css';

function App() {
  return (
    <div className="container">
      Go Visualizer
      <StyledGoboard />
      <Controls />
    </div>
  );
}

export default App;
