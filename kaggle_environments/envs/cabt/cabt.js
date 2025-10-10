
function renderer(context) {
  const step = context.step
  const visList = context.environment.steps[0][0].visualize
  const energyText = "CGRWLPFDM A"

  const info = context.environment.info
  const players = [
    info?.TeamNames?.[0] || "Player 0",
    info?.TeamNames?.[1] || "Player 1"
  ]

  let canvas = context.parent.querySelector("canvas")
  if (!canvas) {
    container = document.createElement("div")
    container.style.position = "relative"
    context.parent.appendChild(container)

    canvas = document.createElement("canvas")
    canvas.width = 750
    canvas.height = 700
    container.appendChild(canvas)

    if (visList) {
      for (let k = 0; k < 2; k++) {
        const button = document.createElement("button")
        button.style.width = "130px"
        button.style.height = "50px"
        button.style.left = k == 0 ? "230px" : "380px"
        button.style.top = "55px"
        button.style.position = "absolute"
        button.style.zIndex = 1
        button.innerHTML = "Open Visualizer<br>" + players[k]
        button.addEventListener("click", (e) => {
          for (let i = 0; i < visList.length; i++) {
            for (let j = 0; j < 2; j++) {
              visList[i].current.players[j].ramainingTime = context.environment.steps[i][j].observation.remainingOverageTime
            }
          }
          visList[0].ps = players

          const input = document.createElement("input")
          input.type = "hidden"
          input.name = "json"
          input.value = JSON.stringify(visList)

          const form = document.createElement("form")
          form.method = "POST"
          form.action = "https://ptcgvis.heroz.jp/Visualizer/Replay/"
          if (info.EpisodeId == null) {
            form.action += k
          } else {
            form.action += info.EpisodeId + "/" + k
          }
          form.target = "_blank"
          form.appendChild(input)

          document.body.appendChild(form)
          form.submit()
        })
        container.appendChild(button)
      }
    } else {
      const ctx = canvas.getContext("2d")
      ctx.strokeStyle = "#ccc"
      ctx.fillStyle = "#fff"
      ctx.font = "30px sans-serif"
      ctx.fillText("No visualize data.", 10, 100)
      const error = context.environment.steps[0][0].error
      if (error) {
        ctx.fillText(error, 10, 150)
      }
    }
  }

  if (visList.length <= step) {
    return
  }
  const vis = visList[step]
  const state = vis.current

  const ctx = canvas.getContext("2d")
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  ctx.strokeStyle = "#ccc"
  ctx.fillStyle = "#fff"
  ctx.lineWidth = 2

  ctx.font = "20px sans-serif"
  if (state.result >= 0) {
    if (state.result == 2) {
      ctx.fillText("Draw", 330, 125)
    } else {
      ctx.fillText(players[state.result] + " Win", 290, 140)
    }
  }

  ctx.font = "12px sans-serif"

  const drawCard = (x, y, card) => {
    ctx.beginPath()
    ctx.rect(x, y, 80, 60)
    ctx.stroke()
    nm = card.name
    nm2 = null
    if (nm.length >= 13) {
      for (let i = 0; i < nm.length; i++) {
        if (nm[i] == " ") {
          nm2 = nm.substring(i + 1)
          nm = nm.substring(0, i)
          break
        }
      }
    }
    ctx.fillText(nm, x + 5, y + 13)
    if (nm2 != null) {
      ctx.fillText(nm2, x + 5, y + 27)
    }
  }
  const drawField = (x, y, card) => {
    drawCard(x, y, card)
    ctx.fillText("HP " + card.hp, x + 5, y + 41)
    energy = ""
    for (let e of card.energies) {
      energy = energy + energyText[e]
    }
    ctx.fillText(energy, x + 5, y + 55)
  }
  const posY = (index, len) => {
    const center = 290
    let height
    if (len <= 8) {
      height = 35 * len
    } else {
      height = 280
    }
    return center + height * (2 * index + 1 - len) / len
  }

  for (let j = 0; j < state.stadium.length; j++) {
    drawCard(330, 420, state.stadium[j])
  }

  for (let i = 0; i < 2; i++) {
    const ps = state.players[i]

    ctx.fillText("Active", i == 0 ? 245 : 425, 270)
    ctx.fillText("Bench", i == 0 ? 145 : 525, 10)
    ctx.fillText("Hand", i == 0 ? 15 : 655, 10)
    ctx.fillText("Deck " + ps.deckCount, i == 0 ? 258 : 438, 165)
    ctx.fillText("Discard " + ps.discard.length, i == 0 ? 245 : 425, 185)
    ctx.fillText("Prize " + ps.prize.length, i == 0 ? 258 : 438, 220)

    for (let j = 0; j < ps.active.length; j++) {
      drawField(i == 0 ? 240 : 420, posY(j, ps.active.length), ps.active[j])
    }
    for (let j = 0; j < ps.bench.length; j++) {
      drawField(i == 0 ? 140 : 520, posY(j, ps.bench.length), ps.bench[j])
    }
    for (let j = 0; j < ps.hand.length; j++) {
      drawCard(i == 0 ? 10 : 650, posY(j, ps.hand.length), ps.hand[j])
    }
  }
}
