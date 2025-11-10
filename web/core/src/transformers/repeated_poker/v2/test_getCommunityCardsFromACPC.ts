import { getCommunityCardsFromACPC } from "./repeatedPokerTransformerUtils.js";

type TestCase = {
  name: string;
  acpcState: string;
  expected?: string;
  expectedError?: string;
};

function runTest({ name, acpcState, expected, expectedError }: TestCase) {
  try {
    const result = getCommunityCardsFromACPC(acpcState);

    if (expectedError) {
      console.error(`❌ ${name}`);
      console.error(
        `   Expected error containing "${expectedError}" but execution succeeded with "${result}"`,
      );
      return;
    }

    if (expected === undefined) {
      console.error(`❌ ${name}`);
      console.error(`   No expected value provided but function returned "${result}"`);
      return;
    }

    const pass = result === expected;

    if (pass) {
      console.log(`✅ ${name}`);
    } else {
      console.error(`❌ ${name}`);
      console.error(`   acpcState: ${acpcState}`);
      console.error(`   expected: "${expected}"`);
      console.error(`   received: "${result}"`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (expectedError && message.includes(expectedError)) {
      console.log(`✅ ${name} (threw expected error)`);
    } else {
      console.error(`❌ ${name}`);
      console.error(`   acpcState: ${acpcState}`);
      if (expectedError) {
        console.error(`   expected error containing: ${expectedError}`);
      }
      console.error(`   received error: ${message}`);
    }
  }
}

const testCases: TestCase[] = [
  {
    name: "preflop - no community cards",
    acpcState: "STATE:0:r5c:AsKs|QhJh",
    expected: "",
  },
  {
    name: "flop dealt",
    acpcState: "STATE:0:r5c:6cKd|AsJc/7hQh6d",
    expected: "7hQh6d",
  },
  {
    name: "turn dealt",
    acpcState: "STATE:0:r5c:6cKd|AsJc/7hQh6d/2h",
    expected: "7hQh6d2h",
  },
  {
    name: "river dealt - full board",
    acpcState: "STATE:0:r5c:6cKd|AsJc/7hQh6d/2h/9s",
    expected: "7hQh6d2h9s",
  },
  {
    name: "empty ACPC state",
    acpcState: "",
    expected: "",
  },
  {
    name: "multi-street with betting",
    acpcState: "STATE:0:r5c/cr11f:6cKd|AsJc/7hQh6d/2h",
    expected: "7hQh6d2h",
  },
  {
    name: "complex full hand",
    acpcState: "STATE:0:r5c/cc/r11c/r122r200c:AsKs|QhJh/7h8h9h/Ts/Js",
    expected: "7h8h9hTsJs",
  },
  {
    name: "malformed state - too few parts",
    acpcState: "STATE:0:r5c",
    expected: "",
  },
  {
    name: "no player delimiter in cards",
    acpcState: "STATE:0:r5c:AsKsQhJh",
    expected: "",
  },
  {
    name: "player cards only, no community cards after delimiter",
    acpcState: "STATE:0:r5c:6cKd|AsJc",
    expected: "",
  },
];

testCases.forEach(runTest);