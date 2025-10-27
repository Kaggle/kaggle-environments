
function renderer(options) {
  // --- Elements and Style Injection ---
  const elements = {
    gameLayout: null,
    pokerTableContainer: null,
    pokerTable: null,
    communityCardsContainer: null,
    potDisplay: null,
    playersContainer: null,
    playerCardAreas: [],
    playerInfoAreas: [],
    dealerButton: null,
    diagnosticHeader: null,
    stepCounter: null,
  };

  const css = `
    @font-face {
      font-family: 'Zeitung Pro';
      src:
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/l?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff2"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/d?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/a?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("opentype");
        font-weight: normal;
        font-style: normal;
    }
    @font-face {
      font-family: 'Zeitung Pro';
      src:
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/l?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff2"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/d?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("woff"),
        url("https://use.typekit.net/af/37ff2c/00000000000000003b9b2a25/27/a?primer=7cdcb44be4a7db8877ffa5c0007b8dd865b3bbc383831fe2ea177f62257a9191&fvd=n7&v=3")
          format("opentype");
        font-weight: bold;
        font-style: normal;
    }

    .poker-renderer-host {
      width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
      font-family: 'Zeitung Pro', sans-serif; background-color: #1C1D20; color: #fff;
      overflow: hidden; padding: 1rem; box-sizing: border-box; position: relative;
    }
    .poker-game-layout { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; position: relative; max-width: 750px; max-height: 750px; }
    .poker-table-container { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; max-width: 750px; max-height: 275px; }
    .poker-table {
      width: clamp(400px, 85vw, 750px); height: clamp(220px, 48vw, 275px);
      background-color: #197631; border-radius: 24px; position: relative;
      display: flex; align-items: center; justify-content: center;
      margin: 0 60px;
    }
    .players-container {
      position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10;
    }
    .player-container {
      position: absolute;
      width: 100%;
      pointer-events: none;
      display: flex;
      flex-direction: column;
    }
    .player-container-0 { bottom: 0; flex-direction: column-reverse; }
    .player-container-1 { top: 0; }
    .player-area-wrapper {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .player-card-area {
      margin: 20px 60px; color: white; text-align: center;
      display: flex; flex-direction: column; justify-content: center; align-items: center;
      min-height: 100px; pointer-events: auto;
    }
    .player-info-area {
      color: white;
      min-width: 180px;
      pointer-events: auto;
      display: flex;
      flex-direction: column;
      justify-content: left;
      align-items: left;
      margin-right: 60px;
    }
    .player-container-0 .player-info-area { flex-direction: column-reverse; }
    .player-name {
      font-size: 32px; font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: white;
      text-align: left;
      padding: 10px 0;
      margin: 0 60px;
    }
    .player-name.winner { color: #FFEB70; }
    .player-stack { font-size: 32px; font-weight: 600; color: #ffffff; margin: 16px 0; display: flex; justify-content: space-between; align-items: center; }
    .player-cards-container { min-height: 70px; display: flex; justify-content: flex-start; align-items:center; gap: 12px; }
    .card {
      display: flex; flex-direction: column; justify-content: space-between; align-items: center;
      width: 80px; height: 112px; border: 2px solid #202124; border-radius: 8px;
      background-color: white; color: black; font-weight: bold; text-align: center; overflow: hidden; position: relative;
      padding: 6px;
    }
    .card-rank { font-family: 'Inter' sans-serif; font-size: 50px; line-height: 1; display: block; align-self: flex-start; }
    .card-suit { width: 50px; height: 50px; display: block; margin-bottom: 2px; }
    .card-suit svg { width: 100%; height: 100%; }
    .card-red .card-rank { color: #B3261E; }
    .card-red .card-suit svg { fill: #B3261E; }
    .card-black .card-rank { color: #000000; }
    .card-black .card-suit svg { fill: #000000; }
    .card-blue .card-rank { color: #0B57D0; }
    .card-blue .card-suit svg { fill: #0B57D0; }
    .card-green .card-rank { color: #146C2E; }
    .card-green .card-suit svg { fill: #146C2E; }
    .card-back {
      background-color: #2b6cb0;
      background-image: linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%, rgba(255,255,255,0.1)),
                        linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.1) 75%, rgba(255,255,255,0.1));
      background-size: 10px 10px; border: 2px solid #63b3ed;
    }
    .card-back .card-rank, .card-back .card-suit { display: none; }
    .card-empty {
      background-color: rgba(255, 255, 255, 0.1);
      border: 2px solid rgba(32, 33, 36, 0.5);
      background-image: none;
    }
    .card-empty .card-rank, .card-empty .card-suit { display: none; }
    .community-cards-area { text-align: center; z-index: 10; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); }
    .community-cards-container { min-height: 75px; display: flex; justify-content: center; align-items:center; margin-bottom: 0.5rem; gap: 12px; }
    .pot-display { font-size: 40px; font-weight: bold; color: #ffffff; margin-bottom: 30px; }
    .bet-display {
      display: inline-block; padding: 10px 20px; border-radius: 12px;
      background-color: #3C4043; color: #ffff;
      font-family: 'Inter' sans-serif; font-size: 1.75rem; font-weigth: 600;
      text-align: center;
      height: 3rem; line-height: 3rem;
      min-width: 200px;
    }
    .blind-indicator { font-size: 0.7rem; color: #a0aec0; margin-top: 3px; }
    .dealer-button {
      width: 36px; height: 36px; background-color: #f0f0f0; color: #333; border-radius: 50%;
      text-align: center; line-height: 36px; font-weight: bold; font-size: 1.5rem; position: absolute;
      border: 3px solid #1EBEFF; box-shadow: 0 1px 3px rgba(0,0,0,0.3); z-index: 15; pointer-events: auto;
    }
    .dealer-button.dealer-player0 { bottom: 110px; }
    .dealer-button.dealer-player1 { top: 110px; }
    .step-counter {
      position: absolute; top: 12px; right: 12px; z-index: 20;
      background-color: rgba(60, 64, 67, 0.9); color: #ffffff;
      padding: 6px 12px; border-radius: 6px;
      font-size: 14px; font-weight: 600;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    @media (max-width: 768px) {
      .bet-display { font-size: 1.5rem; height: 2.2rem; line-height: 2.2rem; min-width: 0;}
      .card { width: 60px; height: 85px; } .card-rank { font-size: 35px; } .card-suit { width: 35px; height: 35px; }
      .community-cards-container { gap: 6px; }
      .player-card-area { min-height: 120px; }
      .player-cards-container { gap: 6px; }
      .player-info-area { min-width: 160px; }
      .poker-game-layout { max-height: 700px; }
      .pot-display { font-size: 35px; margin-bottom: 20px; }
    }
    @media (max-width: 600px) {
      .bet-display { font-size: 20px; height: 40px; line-height: 40px; }
      .card { width: 50px; height: 70px; padding: 2px; } .card-rank { font-size: 32px; } .card-suit { width: 32px; height: 32px; }
      .community-cards-container { gap: 2px; }
      .dealer-button { font-size: 20px; height: 24px; line-height: 24px; width: 24px; }
      .dealer-button.dealer-player0 { bottom: 95px; }
      .dealer-button.dealer-player1 { top: 95px; }
      .player-card-area { min-height: 110px; margin: 0 0 0 40px;}
      .player-cards-container { gap: 2px; }
      .player-info-area { margin-right: 20px; }
      .player-name { font-size: 30px; margin: 0 20px; }
      .player-stack { font-size: 30px; }
      .poker-game-layout { max-height: 600px; }
      .poker-table { width: clamp(300px, 90vw, 600px); height: clamp(160px, 50vw, 200px); margin: 20px; }
      .pot-display { font-size: 30px; margin-bottom: 20px; }
    }
    @media (max-width: 400px) {
      .bet-display { font-size: 15px; height: 30px; line-height: 30px; }
      .card { width: 40px; height: 56px; margin: 0 2px; padding: 2px; } .card-rank { font-size: 25px; } .card-suit { width: 25px; height: 25px; }
      .community-cards-container { gap: 2px; }
      .dealer-button { font-size: 15px; height: 20px; line-height: 20px; width: 20px; }
      .dealer-button.dealer-player0 { bottom: 85px; }
      .dealer-button.dealer-player1 { top: 85px; }
      .player-card-area { margin: 0 0 0 30px;}
      .player-cards-container { gap: 2px; }
      .player-info-area { min-width: 100px; margin-right: 0; }
      .player-name { font-size: 25px; }
      .player-stack { font-size: 15px; }
      .poker-game-layout { max-height: 500px; }
      .poker-table { width: clamp(280px, 95vw, 380px); height: clamp(150px, 55vw, 150px); margin: 0;}
      .pot-display { font-size: 25px; margin-bottom: 15px; }
    }
  `;

  function _injectStyles(passedOptions) {
    if (typeof document === 'undefined' || window.__poker_styles_injected) {
      return;
    }
    const style = document.createElement('style');
    style.textContent = css;
    const parentForStyles = passedOptions && passedOptions.parent ? passedOptions.parent.ownerDocument.head : document.head;
    if (parentForStyles && !parentForStyles.querySelector('style[data-poker-renderer-styles]')) {
      style.setAttribute('data-poker-renderer-styles', 'true');
      parentForStyles.appendChild(style);
    }
    window.__poker_styles_injected = true;
  }

  const suitSVGs = {
    spades: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M31.9017 8.5459L31.9027 8.54688V8.54785L31.9642 8.62988C32.0049 8.68463 32.0647 8.76567 32.1419 8.86914C32.2968 9.07661 32.5214 9.37584 32.7982 9.74316C33.3519 10.4781 34.1164 11.4847 34.9583 12.5713C36.6487 14.7529 38.6314 17.23 39.8587 18.4951C40.5956 19.2546 42.6938 21.1061 45.0882 23.3057C47.4623 25.4866 50.1062 27.9917 51.8763 30.0146C53.659 32.052 54.5809 34.6512 54.9242 37.0439C55.2443 39.2762 55.07 41.3963 54.5648 42.7754L54.4593 43.041L54.4583 43.0439C54.2366 43.5604 53.4581 45.3752 51.889 47.0635C50.312 48.7602 47.9209 50.3437 44.5003 50.3438C41.1459 50.3437 38.4383 49.3111 36.5716 48.2812C35.668 47.7827 34.959 47.2827 34.4662 46.8984C34.6764 47.5682 35.0067 48.3733 35.5287 49.2432C35.8458 49.7716 36.3961 50.2525 37.0941 50.6953C37.7874 51.1352 38.5874 51.513 39.3636 51.8545C40.1218 52.1881 40.8886 52.4987 41.4437 52.7803C41.7223 52.9216 41.9834 53.0734 42.181 53.2383C42.3602 53.3878 42.5999 53.6413 42.5999 54C42.5999 54.3241 42.4172 54.5729 42.2318 54.7422C42.0426 54.9148 41.7911 55.0617 41.5101 55.1895C40.9443 55.4466 40.1512 55.6745 39.1976 55.8652C37.2827 56.2482 34.6237 56.5 31.5999 56.5C28.5752 56.5 25.9176 56.2484 23.9427 55.8662C22.957 55.6754 22.1264 55.4487 21.5003 55.1982C21.1878 55.0732 20.9125 54.9375 20.6908 54.7881C20.4761 54.6434 20.2705 54.4592 20.1527 54.2236L20.0999 54.1182V54C20.0999 53.6414 20.3397 53.3878 20.5189 53.2383C20.7165 53.0734 20.9776 52.9216 21.2562 52.7803C21.8113 52.4987 22.578 52.1881 23.3363 51.8545C24.1124 51.513 24.9125 51.1352 25.6058 50.6953C26.3038 50.2525 26.8541 49.7716 27.1712 49.2432C27.726 48.3186 28.0632 47.467 28.2708 46.7734C28.2466 46.7955 28.2233 46.8199 28.1976 46.8428C27.7563 47.2352 27.101 47.7542 26.2376 48.2725C24.5092 49.3098 21.9429 50.3437 18.5863 50.3438C15.1655 50.3437 12.7737 48.7603 11.1966 47.0635C9.6273 45.3749 8.84884 43.56 8.62728 43.0439L8.62631 43.041C8.04128 41.6783 7.81998 39.4248 8.16146 37.0439C8.50467 34.6513 9.42677 32.052 11.2093 30.0146C12.9793 27.9918 15.6234 25.4865 17.9977 23.3057C20.3921 21.1061 22.4903 19.2546 23.2272 18.4951C24.4545 17.23 26.4372 14.7529 28.1276 12.5713C28.9695 11.4847 29.734 10.4781 30.2877 9.74316C30.5645 9.37584 30.7891 9.07661 30.944 8.86914C31.0212 8.76567 31.081 8.68463 31.1217 8.62988L31.1832 8.54785V8.54688L31.1842 8.5459C31.4355 8.19531 31.8496 8 32.2859 8H32.7999C33.2363 8 33.6504 8.19531 33.9017 8.5459Z"/></svg>',
    hearts: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M31.6667 56.9333L27.8 53.4667C23.3111 49.4222 19.6 45.9333 16.6667 43C13.7333 40.0667 11.4 37.4333 9.66667 35.1C7.93333 32.7667 6.72222 30.6222 6.03333 28.6667C5.34444 26.7111 5 24.7111 5 22.6667C5 18.4889 6.4 15 9.2 12.2C12 9.4 15.4889 8 19.6667 8C21.9778 8 24.1778 8.48889 26.2667 9.46667C28.3556 10.4444 30.1556 11.8222 31.6667 13.6C33.1778 11.8222 34.9778 10.4444 37.0667 9.46667C39.1556 8.48889 41.3556 8 43.6667 8C47.8444 8 51.3333 9.4 54.1333 12.2C56.9333 15 58.3333 18.4889 58.3333 22.6667C58.3333 24.7111 57.9889 26.7111 57.3 28.6667C56.6111 30.6222 55.4 32.7667 53.6667 35.1C51.9333 37.4333 49.6 40.0667 46.6667 43C43.7333 45.9333 40.0222 49.4222 35.5333 53.4667L31.6667 56.9333Z"/></svg>',
    diamonds: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M32 58.3333L8 31.6667L32 5L56 31.6667L32 58.3333Z"/></svg>',
    clubs: '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><path d="M32.7422 8C39.0131 8.00014 44.0965 13.0836 44.0967 19.3545C44.0967 22.3905 42.9028 25.1463 40.9619 27.1836C42.108 26.7945 43.3357 26.5811 44.6133 26.5811C50.8842 26.5813 55.9678 31.6646 55.9678 37.9355C55.9677 44.2065 50.8842 49.2898 44.6133 49.29C40.7767 49.29 37.3866 47.3859 35.3311 44.4727C35.3545 44.6869 35.4 44.9939 35.4873 45.3721C35.6708 46.1669 36.0397 47.2784 36.7832 48.5176C37.1124 49.0661 37.683 49.5639 38.4043 50.0215C39.121 50.4762 39.9477 50.8671 40.749 51.2197C41.5324 51.5644 42.323 51.8854 42.8955 52.1758C43.1826 52.3214 43.4509 52.4767 43.6533 52.6455C43.8375 52.7992 44.0801 53.0572 44.0801 53.4199C44.0799 53.7476 43.8956 54.0007 43.7061 54.1738C43.5126 54.3503 43.2539 54.5014 42.9648 54.6328C42.3825 54.8974 41.5654 55.1324 40.582 55.3291C38.6066 55.7241 35.8618 55.9844 32.7412 55.9844C29.6198 55.9843 26.8772 55.7244 24.8398 55.3301C23.8233 55.1333 22.9671 54.9005 22.3223 54.6426C22.0002 54.5137 21.7169 54.3731 21.4893 54.2197C21.2688 54.0712 21.0593 53.8831 20.9395 53.6436L20.8867 53.5381V53.4199C20.8867 53.0575 21.1294 52.7992 21.3135 52.6455C21.5159 52.4766 21.7851 52.3214 22.0723 52.1758C22.6447 51.8855 23.4346 51.5643 24.2178 51.2197C25.019 50.8672 25.8458 50.4761 26.5625 50.0215C27.2837 49.5639 27.8543 49.066 28.1836 48.5176C28.9271 47.2784 29.297 46.1669 29.4805 45.3721C29.5675 44.9951 29.6113 44.6888 29.6348 44.4746C27.579 47.3866 24.1901 49.29 20.3545 49.29C14.0836 49.2899 9.00003 44.2065 9 37.9355C9 31.6646 14.0835 26.5812 20.3545 26.5811C21.9457 26.5811 23.4603 26.9091 24.835 27.5C22.7097 25.4365 21.3867 22.5506 21.3867 19.3545C21.3869 13.0835 26.4712 8 32.7422 8Z"/></svg>'
  }

  // --- Card Parsing and Rendering ---
  function acpcCardToDisplay(acpcCard) {
    if (!acpcCard || acpcCard.length < 2) return { rank: '?', suit: '', original: acpcCard };
    const rankChar = acpcCard[0].toUpperCase();
    const suitChar = acpcCard[1].toLowerCase();
    const rankMap = { 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A' };
    const suitMap = { 's': 'spades', 'h': 'hearts', 'd': 'diamonds', 'c': 'clubs' };
    const rank = rankMap[rankChar] || rankChar;
    const suit = suitMap[suitChar] || '';
    return { rank, suit, original: acpcCard };
  }

  function createCardElement(cardStr, isHidden = false) {
    const cardDiv = document.createElement('div');
    cardDiv.classList.add('card');
    if (isHidden || !cardStr || cardStr === '?' || cardStr === "??") {
      cardDiv.classList.add('card-back');
    } else {
      const { rank, suit } = acpcCardToDisplay(cardStr);
      const rankSpan = document.createElement('span');
      rankSpan.classList.add('card-rank');
      rankSpan.textContent = rank;
      cardDiv.appendChild(rankSpan);

      const suitSpan = document.createElement('span');
      suitSpan.classList.add('card-suit');

      if (suitSVGs[suit]) {
        suitSpan.innerHTML = suitSVGs[suit];
      }

      cardDiv.appendChild(suitSpan);

      if (suit === 'hearts') cardDiv.classList.add('card-red');
      else if (suit === 'spades') cardDiv.classList.add('card-black');
      else if (suit === 'diamonds') cardDiv.classList.add('card-blue');
      else if (suit === 'clubs') cardDiv.classList.add('card-green');
    }
    return cardDiv;
  }

  // --- Board Parsing and Rendering ---
  function _ensurePokerTableElements(parentElement, passedOptions) {
    if (!parentElement) return false;
    parentElement.innerHTML = '';
    parentElement.classList.add('poker-renderer-host');

    elements.diagnosticHeader = document.createElement('h1');
    elements.diagnosticHeader.id = 'poker-renderer-diagnostic-header';
    elements.diagnosticHeader.textContent = "Poker Table Initialized (Live Data)";
    elements.diagnosticHeader.style.cssText = "color: lime; background-color: black; padding: 5px; font-size: 12px; position: absolute; top: 0px; left: 0px; z-index: 10001; display: none;"; // Hidden by default
    parentElement.appendChild(elements.diagnosticHeader);

    elements.gameLayout = document.createElement('div');
    elements.gameLayout.className = 'poker-game-layout';
    parentElement.appendChild(elements.gameLayout);

    elements.pokerTableContainer = document.createElement('div');
    elements.pokerTableContainer.className = 'poker-table-container';
    elements.gameLayout.appendChild(elements.pokerTableContainer);

    elements.playersContainer = document.createElement('div');
    elements.playersContainer.className = 'players-container';
    elements.gameLayout.appendChild(elements.playersContainer);

    elements.pokerTable = document.createElement('div');
    elements.pokerTable.className = 'poker-table';
    elements.pokerTableContainer.appendChild(elements.pokerTable);

    const communityArea = document.createElement('div');
    communityArea.className = 'community-cards-area';
    elements.pokerTable.appendChild(communityArea);

    elements.potDisplay = document.createElement('div');
    elements.potDisplay.className = 'pot-display';
    communityArea.appendChild(elements.potDisplay);

    elements.communityCardsContainer = document.createElement('div');
    elements.communityCardsContainer.className = 'community-cards-container';
    communityArea.appendChild(elements.communityCardsContainer);

    elements.playerContainers = [];
    elements.playerCardAreas = [];
    elements.playerInfoAreas = [];
    elements.playerNames = [];

    for (let i = 0; i < 2; i++) {
      // Create player container that groups all player elements
      const playerContainer = document.createElement('div');
      playerContainer.className = `player-container player-container-${i}`;
      elements.playersContainer.appendChild(playerContainer);
      elements.playerContainers.push(playerContainer);

      // Player name
      const playerName = document.createElement('div');
      playerName.className = `player-name`;
      playerName.textContent = `Player ${i}`;
      playerContainer.appendChild(playerName);
      elements.playerNames.push(playerName);

      // Create wrapper for card and info areas
      const playerAreaWrapper = document.createElement('div');
      playerAreaWrapper.className = 'player-area-wrapper';
      playerContainer.appendChild(playerAreaWrapper);

      // Card area (left side)
      const playerCardArea = document.createElement('div');
      playerCardArea.className = `player-card-area`;
      playerCardArea.innerHTML = `
        <div class="player-cards-container"></div>
      `;
      playerAreaWrapper.appendChild(playerCardArea);
      elements.playerCardAreas.push(playerCardArea);

      // TODO: Render chip stack
      // Info area (right side)
      const playerInfoArea = document.createElement('div');
      playerInfoArea.className = `player-info-area`;
      playerInfoArea.innerHTML = `
        <div class="player-stack">
            <span class="player-stack-value">0</span>
        </div>
        <div class="bet-display" style="display:none;">Bet : 0</div>
      `;
      playerAreaWrapper.appendChild(playerInfoArea);
      elements.playerInfoAreas.push(playerInfoArea);
    }

    elements.dealerButton = document.createElement('div');
    elements.dealerButton.className = 'dealer-button';
    elements.dealerButton.textContent = 'D';
    elements.dealerButton.style.display = 'none';
    elements.playersContainer.appendChild(elements.dealerButton);

    elements.stepCounter = document.createElement('div');
    elements.stepCounter.className = 'step-counter';
    elements.stepCounter.textContent = 'Standby';
    elements.gameLayout.appendChild(elements.stepCounter);
    return true;
  }

  function _getLastMovesACPC(bettingString, currentPlayer) {
    // We will store all human-readable moves here
    const allMoves = [];

    // Split the action string by street (e.g., ["r5c", "cr11f"])
    const streets = bettingString.split('/');

    // Process each street's actions
    for (let streetIndex = 0; streetIndex < streets.length; streetIndex++) {
      const streetAction = streets[streetIndex];
      let i = 0;

      // Preflop (streetIndex 0), action is "open" due to blinds.
      // Postflop (streetIndex > 0), action is "not open" (first player checks or bets).
      let isAggressiveActionOpen = (streetIndex === 0);

      // 4. Parse the moves within the street
      while (i < streetAction.length) {
        const char = streetAction[i];
        let move = null;

        if (char === 'c') {
          // 'c' (call/check)
          if (isAggressiveActionOpen) {
            move = 'call';
          } else {
            move = 'check';
          }
          isAggressiveActionOpen = false; // 'c' never leaves action open
          i++;
        } else if (char === 'f') {
          // 'f' (fold)
          move = 'fold';
          isAggressiveActionOpen = false; // 'f' ends the hand
          i++;
        } else if (char === 'r') {
          // 'r' (raise/bet)
          let amount = '';
          i++;
          // Continue to parse all digits of the raise amount
          while (i < streetAction.length && streetAction[i] >= '0' && streetAction[i] <= '9') {
            amount += streetAction[i];
            i++;
          }
          move = `raise ${amount}`;
          isAggressiveActionOpen = true; // 'r' always leaves action open
        } else {
          // Should not happen with valid input, but good to prevent infinite loops
          i++;
          continue;
        }

        // 5. Store this move in the history
        if (move) {
          allMoves.push(move);
        }
      }
    }

    // 6. Get the last two moves from our complete list
    const lastMove = allMoves.length > 0 ? allMoves[allMoves.length - 1] : null;
    const secondLastMove = allMoves.length > 1 ? allMoves[allMoves.length - 2] : null;

    const lastMoves = currentPlayer === 0 ? [secondLastMove, lastMove] : [lastMove, secondLastMove];

    return lastMoves;
  }

  function _parseStepHistoryData(universalPokerJSON) {
    const result = {
      cards: [],
      communityCards: '',
      bets: [],
      lastMoves: ['', ''],
      winOdds: [0, 0],
    };

    // Split the string into its main lines
    const lines = universalPokerJSON.acpc_state.trim().split('\n');
    if (lines.length < 2) {
      console.error("Invalid state string format.");
      return result;
    }

    const stateLine = lines[0]; // example: "STATE:0:r5c/cr11c/:6cKd|AsJc/7hQh6d/2c"
    const spentLine = lines[1]; // example: "Spent: [P0: 11  P1: 11  ]"

    // --- Parse the Spent Line ---
    if (spentLine) {
      const p0BetMatch = spentLine.match(/P0:\s*(\d+)/);
      const p1BetMatch = spentLine.match(/P1:\s*(\d+)/);

      const bets = [0, 0];

      if (p0BetMatch) {
        bets[0] = parseInt(p0BetMatch[1], 10);
      }

      if (p1BetMatch) {
        bets[1] = parseInt(p1BetMatch[1], 10);
      }

      result.bets = bets;
    }

    // --- Parse the State Line ---
    if (stateLine) {
      const stateParts = stateLine.split(':');

      // --- Parse Cards ---
      // The card string is always the last part
      const cardString = stateParts[stateParts.length - 1]; // example: "6cKd|AsJc/7hQh6d/2c"

      // Split card string by '/' to separate hand block from board blocks
      const cardSegments = cardString.split('/'); // example: ["6cKd|AsJc", "7hQh6d", "2c"]

      // Parse the first segment (player hands)
      if (cardSegments[0]) {
        const playerHands = cardSegments[0].split('|');
        if (playerHands.length >= 2) {
          // example: "6cKd"
          result.cards = [playerHands[0], playerHands[1]];
        }
      }

      // The rest of the segments are community cards, one per street
      result.communityCards = cardSegments
        .slice(1) // gets all elements AFTER the player hands
        .filter(Boolean) // removes any empty strings (e.g., from a trailing "/")
        .join(''); // joins the remaining segments into a single string

      // --- Parse Betting String --
      // The betting string is everything between the 2nd colon and the last colon.
      // This handles edge cases like "STATE:0:r5c/cr11c/:cards"
      const bettingString = stateParts.slice(2, stateParts.length - 1).join(':');

      if (bettingString) {
        result.lastMoves = _getLastMovesACPC(bettingString, universalPokerJSON.current_player);
      }
    }

    // Parse win odds
    p0WinOdds = Number(universalPokerJSON.odds[0]).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 })
    p1WinOdds = Number(universalPokerJSON.odds[1]).toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 2 })
    result.winOdds = [p0WinOdds, p1WinOdds];

    return result;
  }

  function _getCurrentStepUniversalPokerJSON(options) {
    const { environment, step } = options;

    const agentSteps = environment.info.stateHistory.filter(s => JSON.parse(JSON.parse(s).current_universal_poker_json).current_player !== -1);

    const currentStep = agentSteps[step];
    return JSON.parse(JSON.parse(currentStep).current_universal_poker_json);
  }

  // --- State Parsing ---
  function _parseKagglePokerState(options) {
    const { environment, step } = options;
    const numPlayers = 2;

    // --- Default State ---
    const stateUIData = {
      players: Array(numPlayers).fill(null).map((_, i) => {
        const agentName = environment?.info?.TeamNames?.[i] ||
          `Player ${i}`;
        return {
          id: `player${i}`,
          name: agentName,
          stack: 0,
          cards: [], // Will be filled with nulls or cards
          currentBet: 0,
          isDealer: i === 0,
          isTurn: false,
          reward: null
        };
      }),
      communityCards: [],
      pot: 0,
      isTerminal: false,
      blinds: [1, 2],
      lastMoves: [],
      rawObservation: null, // For debugging
      step: step,
    };

    // --- Step Validation ---
    if (!environment || !environment.steps || !environment.steps[step] || !environment.info) {
      // return default state
      return stateUIData;
    }

    currentStateHistory = JSON.parse(environment.info.stateHistory[step]);
    currentUniversalPokerState = JSON.parse(currentStateHistory.current_universal_poker_json);

    // TODO: Handle the flop phase steps (chance steps)

    currentUniversalPokerJSON = _getCurrentStepUniversalPokerJSON(options);
    currentStepFromStateHistory = _parseStepHistoryData(currentUniversalPokerJSON);

    const currentStepAgents = environment.steps[step];
    if (!currentStepAgents || currentStepAgents.length < numPlayers) {
      return stateUIData;
    }

    const pot_size = currentStepFromStateHistory.bets.reduce((a, b) => a + b, 0);
    const player_contributions = currentStepFromStateHistory.bets;
    const starting_stacks = currentUniversalPokerJSON.starting_stacks;
    const player_hands = [
      currentStepFromStateHistory.cards[0]?.match(/.{1,2}/g) || [],
      currentStepFromStateHistory.cards[1]?.match(/.{1,2}/g) || []
    ];
    const board_cards = currentStepFromStateHistory.communityCards ? currentStepFromStateHistory.communityCards.match(/.{1,2}/g).reverse() : [];

    // TODO: Add odds, best_five_card_hands best_hand_rank_types

    // TODO: Add current player

    const isTerminal = false // TODO: read isTerminal from observation
    stateUIData.isTerminal = isTerminal;
    stateUIData.pot = pot_size || 0;
    stateUIData.communityCards = board_cards || [];
    stateUIData.lastMoves = currentStepFromStateHistory.lastMoves;
    stateUIData.blinds = currentUniversalPokerState.blinds;

    // --- Update Players ---
    for (let i = 0; i < numPlayers; i++) {
      const pData = stateUIData.players[i];
      const contribution = player_contributions ? player_contributions[i] : 0;
      const startStack = starting_stacks ? starting_stacks[i] : 0;

      pData.currentBet = contribution;
      pData.stack = startStack - contribution;
      pData.cards = (player_hands[i] || []).map(c => c === "??" ? null : c);
      pData.isTurn = currentUniversalPokerState.current_player === i;
      pData.isDealer = currentUniversalPokerState.blinds[i] === 1; // infer dealer from small blind

      if (isTerminal) {
        const reward = environment.rewards ? environment.rewards[i] : null;
        pData.reward = reward;
        if (reward > 0) {
          pData.name = `${pData.name} wins 🎉`;
          pData.isWinner = true;
          pData.status = null;
        } else {
          pData.status = null;
        }
      } else if (pData.stack === 0 && pData.currentBet > 0) {
        pData.status = "All-in";
      }
    }

    return stateUIData;
  }


  function _renderPokerTableUI(data, passedOptions) {
    if (!elements.pokerTable || !data) return;
    const { players, communityCards, pot, isTerminal, step } = data;

    // Update step counter
    if (elements.stepCounter && step !== undefined) {
      elements.stepCounter.textContent = `Step: ${step}`;
    }

    if (elements.diagnosticHeader && data.rawObservation) {
      // Optional: Show diagnostics for debugging
      // elements.diagnosticHeader.textContent = `[${passedOptions.step}] P_TURN:${data.rawObservation.current_player} POT:${data.pot}`;
      // elements.diagnosticHeader.style.display = 'block';
    }

    elements.communityCardsContainer.innerHTML = '';
    // Always show 5 slots for the river
    // Display cards left to right, with empty slots at the end
    const numCommunityCards = 5;
    const numCards = communityCards ? communityCards.length : 0;

    // Since the 4th and 5th street cards are appended to the communityCards array, we need to
    // reverse it so that the added cards are put at the end of the display area on the board.
    if (communityCards) communityCards.reverse();

    // Add actual cards
    for (let i = 0; i < numCards; i++) {
      elements.communityCardsContainer.appendChild(createCardElement(communityCards[i]));
    }

    // Fill remaining slots with empty cards
    for (let i = numCards; i < numCommunityCards; i++) {
      const emptyCard = document.createElement('div');
      emptyCard.classList.add('card', 'card-empty');
      elements.communityCardsContainer.appendChild(emptyCard);
    }

    elements.potDisplay.textContent = `Pot : ${pot}`;

    players.forEach((playerData, index) => {
      const playerNameElement = elements.playerNames[index];
      if (playerNameElement) {
        const playerNameText = playerData.isTurn && !isTerminal ? `${playerData.name} responding...` : playerData.name;
        playerNameElement.textContent = playerNameText;

        // Add winner class if player won
        if (playerData.isWinner) {
          playerNameElement.classList.add('winner');
        } else {
          playerNameElement.classList.remove('winner');
        }
      }

      // Update card area (left side)
      const playerCardArea = elements.playerCardAreas[index];
      if (playerCardArea) {

        const playerCardsContainer = playerCardArea.querySelector('.player-cards-container');
        playerCardsContainer.innerHTML = '';

        // In heads-up, we show both hands at the end.
        const showCards = isTerminal || (playerData.cards && !playerData.cards.includes(null));

        (playerData.cards || [null, null]).forEach(cardStr => {
          playerCardsContainer.appendChild(createCardElement(cardStr, !showCards && cardStr !== null));
        });
      }

      // Update info area (right side)
      const playerInfoArea = elements.playerInfoAreas[index];
      if (playerInfoArea) {
        playerInfoArea.querySelector('.player-stack-value').textContent = `${playerData.stack}`;

        const betDisplay = playerInfoArea.querySelector('.bet-display');
        if (playerData.currentBet > 0) {
          if (data.lastMoves[index]) {
            betDisplay.textContent = data.lastMoves[index];
          }
          else {
            if (playerData.isDealer) {
              betDisplay.textContent = 'Small Blind';
            } else {
              betDisplay.textContent = 'Big Blind';
            }
          }
          betDisplay.style.display = 'block';
        } else {
          betDisplay.style.display = 'none';
        }
      }
    });

    const dealerPlayerIndex = players.findIndex(p => p.isDealer);
    if (elements.dealerButton) {
      if (dealerPlayerIndex !== -1) {
        elements.dealerButton.style.display = 'block';
        // Remove previous dealer class
        elements.dealerButton.classList.remove('dealer-player0', 'dealer-player1');
        // Add new dealer class based on player index
        elements.dealerButton.classList.add(`dealer-player${dealerPlayerIndex}`);
      } else {
        elements.dealerButton.style.display = 'none';
      }
    }
  }


  // --- MAIN EXECUTION LOGIC ---
  const { parent } = options;
  if (!parent) {
    console.error("Renderer: Parent element not provided.");
    return;
  }

  _injectStyles(options);

  if (!_ensurePokerTableElements(parent, options)) {
    console.error("Renderer: Failed to ensure poker table elements.");
    parent.innerHTML = '<p style="color:red;">Error: Could not create poker table structure.</p>';
    return;
  }

  const uiData = _parseKagglePokerState(options);
  _renderPokerTableUI(uiData, options);
}