import { __testing } from "./repeatedPokerTransformer.js";

type TestCase = {
  name: string;
  bettingString: string;
  expected?: string[];
  expectedError?: string;
};

function runTest(
  { name, bettingString, expected, expectedError }: TestCase,
  transformer: (bettingString: string) => string[],
) {
  try {
    const result = transformer(bettingString);

    if (expectedError) {
      console.error(`❌ ${name}`);
      console.error(
        `   Expected error containing "${expectedError}" but execution succeeded with ${JSON.stringify(
          result,
        )}`,
      );
      return;
    }

    if (!expected) {
      console.error(`❌ ${name}`);
      console.error(
        `   No expected moves provided but transformer returned ${JSON.stringify(result)}`,
      );
      return;
    }

    const pass =
      result.length === expected.length &&
      result.every((move, idx) => move === expected[idx]);

    if (pass) {
      console.log(`✅ ${name}`);
    } else {
      console.error(`❌ ${name}`);
      console.error(`   bettingString: ${bettingString}`);
      console.error(`   expected: ${JSON.stringify(expected)}`);
      console.error(`   received: ${JSON.stringify(result)}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (expectedError && message.includes(expectedError)) {
      console.log(`✅ ${name} (threw expected error)`);
    } else {
      console.error(`❌ ${name}`);
      console.error(`   bettingString: ${bettingString}`);
      if (expectedError) {
        console.error(`   expected error containing: ${expectedError}`);
      }
      console.error(`   received error: ${message}`);
    }
  }
}

const testCases: TestCase[] = [
  {
    name: "preflop raise-call sequence",
    bettingString: "r5c",
    expected: ["r5", "c"],
  },
  {
    name: "multi-street action with fold",
    bettingString: "r5c/cr11f",
    expected: ["r5", "c", "c", "r11", "f"],
  },
  {
    name: "check-check through two streets",
    bettingString: "cc/cc",
    expected: ["c", "c", "c", "c"],
  },
  {
    name: "empty betting string",
    bettingString: "",
    expected: [""].filter(Boolean),
  },
];

const readableTestCases: TestCase[] = [
  {
    name: "preflop raise and call",
    bettingString: "r4c",
    expected: ["Raise 4", "Call 2"],
  },
  {
    name: "small blind call then big blind raise",
    bettingString: "cr6c",
    expected: ["Call 1", "Raise 6", "Call 4"],
  },
  {
    name: "flop betting with call",
    bettingString: "r4c/cr8c",
    expected: ["Raise 4", "Call 2", "Check", "Bet 4", "Call 4"],
  },
  {
    name: "immediate check back",
    bettingString: "r4c/c",
    expected: ["Raise 4", "Call 2", "Check"],
  },
  {
    name: "multi-raise same street",
    bettingString: "r5r11",
    expected: ["Raise 5", "Raise 11"],
  },
  {
    name: "invalid raise target throws",
    bettingString: "r4r2",
    expectedError: "Invalid raise target",
  },
  {
    name: "unknown betting token throws",
    bettingString: "r4x",
    expectedError: "Unknown betting token",
  },
  {
    name: "empty readable betting string",
    bettingString: "",
    expected: [],
  },
  {
    name: "full hand example 1",
    bettingString: "r5c/cc/r11c/r122r200c",
    expected: ["Raise 5", "Call 3", "Check", "Check", "Bet 6", "Call 6", "Bet 111", "Raise 189", "Call 78"],
  },
  {
    name: "full hand example 2",
    bettingString: "r10r20r30c/r32r34r44r54c/r56r80c/cr100r150r200c",
    expected: ["Raise 10","Raise 20","Raise 30","Call 10","Bet 2","Raise 4","Raise 14","Raise 24","Call 10","Bet 2","Raise 26","Call 24","Check","Bet 20","Raise 70","Raise 120","Call 50"],
  },
];

const { _getMovesFromBettingStringACPC, _getReadableMovesFromBettingStringACPC } =
  __testing;

testCases.forEach((test) =>
  runTest(test, _getMovesFromBettingStringACPC),
);
readableTestCases.forEach((test) =>
  runTest(test, _getReadableMovesFromBettingStringACPC),
);
