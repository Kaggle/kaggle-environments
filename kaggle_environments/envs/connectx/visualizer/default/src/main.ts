import { Player, PreactAdapter } from "@kaggle-environments/core";
import { Renderer } from "./renderer";
import "./style.css";

const app = document.getElementById("app");
if (!app) {
  throw new Error("Could not find app element");
}

const adapter = new PreactAdapter(Renderer);
new Player(app, adapter);
