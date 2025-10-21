
function renderer(context) {
  const { parent, environment, step } = context;
  parent.innerHTML = '';  // Clear previous rendering

  const currentStepData = environment.steps[step];
  if (!currentStepData) {
    parent.textContent = "Waiting for step data...";
    return;
  }
  const agentObsIndex = 0
  let obsString = "Observation not available for this step.";
  let title = `Step: ${step}`;

  if (environment.configuration && environment.configuration.openSpielGameName) {
    title = `${environment.configuration.openSpielGameName} - Step: ${step}`;
  }

  currentStep = JSON.parse(environment.info.stateHistory[step]);
  obsString = environment.info.stateHistory[step];

  if (step === 0 && environment.steps[0] && environment.steps[0][agentObsIndex] &&
    environment.steps[0][agentObsIndex].observation &&
    typeof environment.steps[0][agentObsIndex].observation.observationString === 'string') {
    obsString = environment.steps[0][agentObsIndex].observation.observationString;
  }

  const pre = document.createElement("pre");
  pre.style.fontFamily = "monospace";
  pre.style.margin = "10px";
  pre.style.border = "1px solid #ccc";
  pre.style.padding = "10px";
  pre.style.backgroundColor = "#f9f9f9";
  pre.style.whiteSpace = "pre-wrap";
  pre.style.wordBreak = "break-all";

  pre.textContent = `${title}\\n\\n${obsString}`;
  parent.appendChild(pre);
}