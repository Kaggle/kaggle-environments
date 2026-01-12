import { getReadableActionsFromACPC } from "./repeatedPokerTransformerUtils.js";

type TestCase = {
  name: string;
  bettingString: string;
  expected?: string[];
  expectedError?: string;
};

const buildAcpcState = (bettingString: string): string =>
  bettingString ? `STATE:0:${bettingString}:AsKs|QhJh` : `STATE:0::AsKs|QhJh`;

function runTest({ name, bettingString, expected, expectedError }: TestCase) {
  try {
    const result = getReadableActionsFromACPC(buildAcpcState(bettingString));

    if (expectedError) {
      console.error(`❌ ${name}`);
      console.error(
        `   Expected error containing "${expectedError}" but execution succeeded with ${JSON.stringify(result)}`,
      );
      return;
    }

    if (!expected) {
      console.error(`❌ ${name}`);
      console.error(`   No expected moves provided but transformer returned ${JSON.stringify(result)}`);
      return;
    }

    const pass = result.length === expected.length && result.every((move, idx) => move === expected[idx]);

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
    expected: ["Raise 5", "Call 3"],
  },
  {
    name: "multi-street action with fold",
    bettingString: "r5c/cr11f",
    expected: ["Raise 5", "Call 3", "Check", "Bet 6", "Fold"],
  },
  {
    name: "check-check through two streets",
    bettingString: "cc/cc",
    expected: ["Call 1", "Check", "Check", "Check"],
  },
  {
    name: "empty betting string",
    bettingString: "",
    expected: [],
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
    name: "full hand example",
    bettingString: "r5c/cc/r11c/r122r200c",
    expected: ["Raise 5", "Call 3", "Check", "Check", "Bet 6", "Call 6", "Bet 111", "Raise 189", "Call 78"],
  },
];

testCases.forEach(runTest);
