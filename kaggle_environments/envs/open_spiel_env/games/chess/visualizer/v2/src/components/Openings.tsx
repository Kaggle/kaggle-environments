import { useEffect } from 'react';
import useChessStore from '../stores/useChessStore';

export default function Openings() {
  const chess = useChessStore((state) => state.chess);

  useEffect(() => {
  const openings = [
    // --- FLANK & UNUSUAL OPENINGS ---
    { name: 'Anderssen Opening', fen: 'rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1' },
    { name: 'Bird\'s Opening', fen: 'rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 1' },
    { name: 'English Opening', fen: 'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1' },
    { name: 'Grob Opening', fen: 'rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq g3 0 1' },
    { name: 'Nimzowitsch-Larsen Attack', fen: 'rnbqkbnr/pppppppp/8/8/8/1P6/P1PPPPPP/RNBQKBNR b KQkq - 0 1' },
    { name: 'Polish Opening', fen: 'rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq b3 0 1' },
    { name: 'Reti Opening', fen: 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1' },
    { name: 'Van Geet Opening', fen: 'rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1' },
    { name: 'Van\'t Kruijs Opening', fen: 'rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1' },
    { name: 'Mieses Opening', fen: 'rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1' },

    // --- KING'S PAWN (1. e4) & VARIATIONS ---
    { name: 'King\'s Pawn Opening', fen: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1' },
    { name: 'Alekhine\'s Defense', fen: 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 0 1' },
    { name: 'Caro-Kann Defense', fen: 'rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2' },
    { name: 'French Defense', fen: 'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2' },
    { name: 'French Defense: Advance Variation', fen: 'rnbqkbnr/pp1p1ppp/4p3/2pPP3/8/8/PPP2PPP/RNBQKBNR b KQkq - 0 3' },
    { name: 'French Defense: Winawer Variation', fen: 'rnbqk1nr/pp3ppp/4p3/2ppP3/3P4/2P5/P1P2PPP/R1BQKBNR w KQkq - 0 7' },
    { name: 'French Defense: Tarrasch Variation', fen: 'rnbqkbnr/pp1p1ppp/4p3/8/3PP3/8/PP1N1PPP/R1BQKBNR b KQkq - 1 3' },
    { name: 'Italian Game', fen: 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3' },
    { name: 'King\'s Gambit', fen: 'rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3 0 2' },
    { name: 'Petrov\'s Defense', fen: 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 3 2' },
    { name: 'Pirc Defense', fen: 'rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2' },
    { name: 'Ruy Lopez', fen: 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3' },
    { name: 'Ruy Lopez: Marshall Attack', fen: 'r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 b - - 0 9' },
    { name: 'Ruy Lopez: Berlin Defense', fen: 'r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4' },
    { name: 'Scandinavian Defense', fen: 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2' },
    { name: 'Scotch Game', fen: 'r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3 0 3' },
    { name: 'Vienna Game', fen: 'rnbqkbnr/pppp1ppp/8/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 2' },
    { name: 'Bongcloud Attack', fen: 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2' },
    { name: 'Stafford Gambit', fen: 'r1bqkb1r/pppp1ppp/2n5/4P3/2B1n3/5N2/PPP2PPP/RNBQK2R b KQkq - 0 5' },

    // --- SICILIAN DEFENSE FAMILY ---
    { name: 'Sicilian Defense', fen: 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2' },
    { name: 'Sicilian Defense: Najdorf Variation', fen: 'rnbqkb1r/1p1ppppp/p4n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 6' },
    { name: 'Sicilian Defense: Dragon Variation', fen: 'rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 6' },
    { name: 'Sicilian Defense: Alapin Variation', fen: 'rnbqkbnr/pp1ppppp/8/2p5/4P3/2P5/PP1P1PPP/RNBQKBNR b KQkq - 0 2' },
    { name: 'Sicilian Defense: Smith-Morra Gambit', fen: 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2' },
    { name: 'Sicilian Defense: Accelerated Dragon', fen: 'r1bqkbnr/pp1ppp1p/2n3p1/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5' },

    // --- QUEEN'S PAWN (1. d4) & VARIATIONS ---
    { name: 'Queen\'s Pawn Opening', fen: 'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1' },
    { name: 'Queen\'s Gambit', fen: 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2' },
    { name: 'Queen\'s Gambit Accepted', fen: 'rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3' },
    { name: 'Slav Defense', fen: 'rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3' },
    { name: 'London System', fen: 'rnbqkbnr/pp1ppppp/8/2p5/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 1 2' },
    { name: 'Catalan Opening', fen: 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3' },
    { name: 'Dutch Defense', fen: 'rnbqkbnr/pppppppp/8/8/3P1P2/8/PPP1P1PP/RNBQKBNR b KQkq - 0 1' },
    { name: 'Budapest Gambit', fen: 'rnbqkbnr/pppp1ppp/8/4p3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 2' },

    // --- INDIAN DEFENSES ---
    { name: 'King\'s Indian Defense', fen: 'rnbqkbnr/ppp1pp1p/6p1/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3' },
    { name: 'Nimzo-Indian Defense', fen: 'r1bqkbnr/pppp1ppp/2n1p3/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 3' },
    { name: 'Grünfeld Defense', fen: 'rnbqkbnr/ppp1pp1p/6p1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 3' },
    { name: 'Bogo-Indian Defense', fen: 'rnbqk2r/pppp1ppp/4pn2/8/2PP4/5N2/PP1BPPPP/RN1QKB1R b KQkq - 2 4' },
    { name: 'Old Indian Defense', fen: 'rnbqkbnr/ppp1pppp/3p4/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 2' }
  ];

    const opening = openings.find((opening) => opening.fen === chess.fen());

    if (opening) console.log(`*** ${opening.name} ***`);
  }, [chess]);

  return null;
}
