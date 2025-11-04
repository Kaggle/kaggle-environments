import { Player, PreactAdapter } from "@kaggle-environments/core";
import { Renderer } from "./renderer";
import "./style.css";

const app = document.getElementById("app");
if (!app) {
  throw new Error("Could not find app element");
}

// TODO - fix this when we figure out a global format
const adapter = new PreactAdapter(Renderer as any);
new Player(app, adapter);
