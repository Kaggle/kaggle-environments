async function renderer(context) {
    const {
        act,
        agents,
        environment,
        frame,
        height = 400,
        interactive,
        isInteractive,
        parent,
        step,
        update,
        width = 400,
    } = context;

    function createContainer(parent, name, weight, direction) {
        let container = parent.querySelector("." + name);
        if (!container) {
            container = document.createElement("div");
            container.className = name;
            container.style.display = 'flex';
            container.style.flex = weight + ' 0 0';
            container.style.flexFlow = direction + ' nowrap';
            parent.appendChild(container);
        }

        return container;
    }

    function createRow(parent, name, weight) {
        return createContainer(parent, name, weight, 'row');
    }

    function createColumn(parent, name, weight) {
        return createContainer(parent, name, weight, 'column');
    }

    // Create text panes in a particular column
    function createPane(parent, name, weight, text) {
        let pane = parent.querySelector("." + name);
        if (!pane) {
            pane = document.createElement("textarea");
            pane.className = name;
            pane.readOnly = true;
            pane.style.resize = 'none';
            pane.style.width = '100%';
            pane.style.height = '100%';

            const label = document.createElement("label");
            label.innerHTML = name;
            label.for = pane;
            label.style.color = "white";

            const div = document.createElement("div");
            div.style.display = 'flex';
            div.style.flex = weight + ' 0 0';
            div.style.flexFlow = 'column nowrap';
            div.style.margin = "4px";

            div.appendChild(label);
            div.appendChild(pane);
            parent.appendChild(div);
        }

        pane.value = JSON.stringify(text, null, 2);
        return pane;
    }

    // Extract
    const state = environment.steps[step];
    const observations = state.map(({ observation }) => observation);
    const actions = state.map(({ action }) => action);
    const rewards = state.map(({ reward }) => reward);
    const statuses = state.map(({ status }) => status);

    // Create the main container
    const panes = createRow(parent, "panes", 1);
    panes.style.width = '100%';
    panes.style.height = '100%';
    
    createPane(panes, 'Observations', 3, observations)
    
    const rightColumn = createColumn(panes, 'rightColumn', 3);
    createPane(rightColumn, 'Configuration', 2, environment.configuration);
    
    const bottomRightRow = createRow(rightColumn, 'bottomRightRow', 1);
    createPane(bottomRightRow, 'Actions', 1, actions)
    createPane(bottomRightRow, 'Rewards', 1, rewards)
    createPane(bottomRightRow, 'Statuses', 1, statuses)
}
