import { useRive, Layout, Alignment, Fit } from "@rive-app/react-canvas";

export default function Hero() {
  const {RiveComponent} = useRive({
    src: 'kaggle_knight.riv',
    stateMachines: `State Machine 1`,
    autoplay: true,
    useOffscreenRenderer: true,
    autoBind:true,
    layout: new Layout({ fit:Fit.Contain, alignment: Alignment.Center }), 
  })

  return (
    <div id="hero">
      <RiveComponent width="480" height="600"/>
    </div>
  );
}
