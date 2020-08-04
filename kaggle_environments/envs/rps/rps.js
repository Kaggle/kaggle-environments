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

    weapon_name = ["Rock","Paper","Scissors","Spock","Lizard","Airplane","Sun","Moon","Camera","Grass","Fire","Film","Spanner","Toilet","School","Air","Death","Planet","Curse","Guitar","Lock","Bowl","Pickaxe","Cup","Peace","Beer","Computer","Rain","Castle","Water","Snake","TV","Blood","Rainbow","Porcupine","UFO","Eagle","Alien","Monkey","Prayer","King","Mountain","Queen","Satan","Wizard","Dragon","Mermaid","Diamond","Police","Trophy","Woman","Money","Baby","Devil","Man","Link","Home","Video Game","Train","Math","Car","Robot","Noise","Heart","Bicycle","Electricity","Tree","Lightning","Potato","Ghost","Duck","Power","Wolf","Microscope","Cat","Nuke","Chicken","Cloud","Fish","Truck","Spider","Helicopter","Bee","Bomb","Brain","Tornado","Community","Sand","Zombie","Pit","Bank","Chain","Vampire","Gun","Bath","Law","Monument","Baloon","Pancake","Sword","Book"]
    weapon_emoji = ["👊🏽","📄","✂️","🖖","🦎","✈️","☀️","🌙","📷","🌱","🔥","🎥","🔧","🚽","🏫","💨","☠","🌎","🥀","🎸","🔒","🥣","⛏️","☕","🕊️","🍺","💻","🌧️","🏰","💧","🐍","📺","💉","🌈","🦔","🛸","🦅","👽","🐒","🙏🏽","🤴🏼","🏔️","👸🏽","😈","🧙🏼‍♂️","🐉","🧜🏽‍♀️","💎","👮🏽‍♀️","🏆","👩🏻","💰","👶🏽","👹","👨🏾","🔗","🏠","🎮","🚂","🔢","🚗","🤖","🔔","❤️","🚲","💡","🌲","⚡","🥔","👻","🦆","🔋","🐺","🔬","🐈","☢️","🐓","☁️","🐟","🚚","🕷️","🚁","🐝","💣","🧠","🌪️","👥","🏖️","🧟‍♂️","🕳️","🏦","⛓️","🧛🏽‍♂️","🔫","🛁","⚖️","🏛️","🎈","🥞","🗡️","📖"]

    // ------------------------------------------------------------------------------------//

    // Common Dimensions.
    const canvasSize = Math.min(height, width);
    const unit = 8;
    const offset = canvasSize > 400 ? canvasSize * 0.1 : unit / 2;
    const cellSize = (canvasSize - offset * 2) / 3;

    // Canvas Setup.
    let canvas = parent.querySelector("canvas");
    if (!canvas) {
        canvas = document.createElement("canvas");
        parent.appendChild(canvas);

        if (interactive) {
            canvas.addEventListener("click", evt => {
                if (!isInteractive()) return;
                const rect = evt.target.getBoundingClientRect();
                const x = evt.clientX - rect.left - offset;
                const y = evt.clientY - rect.top - offset;
                act(Math.floor(x / cellSize) + Math.floor(y / cellSize) * 3);
            });
        }
    }
    canvas.style.cursor = isInteractive() ? "pointer" : "default";

    // Canvas setup and reset.
    let c = canvas.getContext("2d");
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    c.clearRect(0, 0, canvas.width, canvas.height);

    // ------------------------------------------------------------------------------------//

    test = environment.steps[step][0].observation.results[0]

    if (test !=null) {

        p1moves = environment.steps[step][0].observation.p1_moves;
        p2moves = environment.steps[step][0].observation.p2_moves;
        results = environment.steps[step][0].observation.results;

        p1move = p1moves[p1moves.length - 1];
        p2move = p2moves[p2moves.length - 1];
        result = results[results.length - 1];

        p1score = 0;
            for(i = 0; i < results.length; ++i){
                if(results[i] == 1)
                p1score++;
        }

        p2score = 0;
            for(i = 0; i < results.length; ++i){
                if(results[i] == 2)
                p2score++;
        }

        ties = 0;
            for(i = 0; i < results.length; ++i){
                if(results[i] == 0)
                ties++;
        }

        ctx = canvas.getContext("2d");

        // Player Row

        ctx.font = "20px sans-serif";
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText("Player 1",0,20)
        ctx.fillText("Player 2",170,20)

        // Emoji Row

        ctx.font = "60px sans-serif";
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText(weapon_emoji[p1move - 1], 0,100);
        ctx.fillText("vs", 90,100);
        ctx.fillText(weapon_emoji[p2move - 1], 180,100);

        // Result Row

        ctx.font = "20px sans-serif";
        ctx.fillStyle = "#FFFFFF";

        if (result == 1) {
            ctx.fillText("Win", 15,150);
        }

        if (result == 2) {
            ctx.fillText("Win", 195,150);
        }

        if (result == 0) {
            ctx.fillText("Tie", 105,150);
        }

        // Score Row
        ctx.font = "20px sans-serif";
        ctx.fillStyle = "#FFFFFF";
        ctx.fillText(p1score, 20,200);
        ctx.fillText(ties, 110,200);
        ctx.fillText(p2score, 205,200);
    }
}