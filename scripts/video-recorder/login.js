/* eslint-disable no-undef */

// Credentials for a Kaggle account, replace with your own and DO NOT MERGE.
const creds = {
  email: 'USERNAME',
  password: 'PASSWORD',
};

const BASE_URL = 'https://www.kaggle.com';

async function login(page) {
  // Sign In
  const loginUrl = `${BASE_URL}/account/login?phase=emailSignIn`;
  console.log(`Attempting to navigate to: ${loginUrl}`);

  // Try to load the login page in a retry loop because sometimes tests start before
  // the backend (HttpRoute) is ready to serve the requests. In that case the backend
  // returns a 404 response:
  // "response 404 (backend NotFound), service rules for the path non-existent"
  const maxRetries = 100;
  const retryDelaySeconds = 3;
  let response = null;

  for (let i = 0; i < maxRetries; i++) {
    try {
      response = await page.goto(loginUrl, {
        waitUntil: 'domcontentloaded',
      });
      // If we get a response and it's not a 4xx or 5xx, we're good.
      const status = response ? response.status() : -1;
      if (status < 400) {
        console.log(`Successfully navigated to login page with status: ${status}`);
        break;
      }
      console.log(`Attempt ${i + 1} failed with status ${status}. Retrying in ${retryDelaySeconds}s...`);
    } catch (error) {
      console.log(`Attempt ${i + 1} failed with error: ${error}. Retrying in ${retryDelaySeconds}s...`);
    }

    if (i < maxRetries - 1) {
      await page.waitForTimeout(retryDelaySeconds * 1000);
    } else {
      // If we've exhausted retries, dump the last known state and throw.
      console.log('Final page content on failure: ', await page.content());
      throw new Error(`Failed to load login page at ${loginUrl} after ${maxRetries} attempts.`);
    }
  }

  await page.getByPlaceholder('Enter your email address or username').fill(creds.email);
  await page.getByPlaceholder('Enter password').fill(creds.password);

  // Use promise.all to prevent race conditions. Wait for the login response, then click the sign in button.
  await Promise.all([
    page.waitForResponse(`${BASE_URL}/api/i/users.LegacyUsersService/EmailSignIn`),
    await page.click('button:has-text("Sign In")'),
  ]);
  await page.waitForURL(`${BASE_URL}`);
  await page.context().storageState({ path: 'logged-in-state.json' });
}

module.exports = { login };
