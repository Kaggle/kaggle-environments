
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

    // Try to get obs_string from game_master of current step
    if (currentStepData[agentObsIndex] && 
        currentStepData[agentObsIndex].observation && 
        typeof currentStepData[agentObsIndex].observation.observationString === 'string') {
        obsString = currentStepData[agentObsIndex].observation.observationString;
    } 
    // Fallback to initial step if current is unavailable (e.g. very first render call)
    else if (step === 0 && environment.steps[0] && environment.steps[0][agentObsIndex] && 
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