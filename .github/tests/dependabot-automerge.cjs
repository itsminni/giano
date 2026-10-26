const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const assert = require('node:assert/strict');
const { test } = require('node:test');

// Exercise the exact privileged workflow code without network calls or tokens.
const workflow = readFileSync(resolve(__dirname, '../workflows/dependabot-automerge.yml'), 'utf8')
  .replace(/\r\n/g, '\n');
const script = workflow.match(/          script: \|\n([\s\S]*?)\n      - name: Enable auto-merge/)[1]
  .split('\n').map(line => line.replace(/^ {12}/, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const evaluate = new AsyncFunction('github', 'context', 'core', script);
const required = ['quality (ubuntu-latest)', 'quality (windows-latest)', 'website'];

function baseline() {
  return {
    updates: [{ updateType: 'version-update:semver-patch' }, { updateType: 'version-update:semver-minor' }],
    repository: {
      autoMergeAllowed: true, squashMergeAllowed: true,
      pullRequest: {
        state: 'OPEN', isDraft: false, headRefOid: 'expected-head', baseRefName: 'main',
        author: { login: 'dependabot[bot]' }, headRepository: { nameWithOwner: 'itsminni/giano' },
      },
    },
    rules: [{
      type: 'required_status_checks',
      parameters: {
        strict_required_status_checks_policy: true,
        required_status_checks: required.map(context => ({ context, integration_id: 15368 })),
      },
    }],
    files: [{ status: 'modified', filename: 'website/giano-weather-vision/bun.lock' }],
  };
}

async function run(state) {
  const outputs = {};
  const failures = [];
  const previous = process.env.UPDATED_DEPENDENCIES;
  process.env.UPDATED_DEPENDENCIES = state.raw ?? JSON.stringify(state.updates);
  const listFiles = () => { throw new Error('Calls must use pagination'); };
  const github = {
    rest: { pulls: { listFiles } },
    graphql: async query => {
      // A future edit must not restore the administrative GraphQL field.
      assert(!query.includes('branchProtectionRule'));
      return { repository: state.repository };
    },
    paginate: async (route, args) => {
      assert.equal(args.owner, 'itsminni');
      assert.equal(args.repo, 'giano');
      if (typeof route === 'string') {
        assert.equal(route, 'GET /repos/{owner}/{repo}/rules/branches/{branch}');
        assert.equal(args.branch, 'main');
        return state.rules;
      }
      assert.equal(route, listFiles);
      assert.equal(args.pull_number, 42);
      return state.files;
    },
  };
  try {
    await evaluate(github, {
      repo: { owner: 'itsminni', repo: 'giano' },
      payload: { pull_request: { number: 42, head: { sha: 'expected-head' } } },
    }, {
      info: () => {},
      setFailed: message => failures.push(message),
      setOutput: (name, value) => { outputs[name] = value; },
    });
  } catch (error) {
    failures.push(error.message);
  } finally {
    if (previous === undefined) delete process.env.UPDATED_DEPENDENCIES;
    else process.env.UPDATED_DEPENDENCIES = previous;
  }
  return { outputs, failures };
}

const cases = [
  ['patch/minor lock update', () => {}, true],
  ['existing workflow version update', s => { s.files = [{ status: 'modified', filename: '.github/workflows/ci.yml' }]; }, true],
  ['major update', s => { s.updates.push({ updateType: 'version-update:semver-major' }); }, false],
  ['unclassified update', s => { s.updates.push({ updateType: '' }); }, false],
  ['empty metadata', s => { s.updates = []; }, false],
  ['malformed metadata', s => { s.raw = '{invalid'; }, false],
  ['wrong metadata shape', s => { s.updates = {}; }, false],
  ['auto-merge disabled', s => { s.repository.autoMergeAllowed = false; }, false],
  ['squash disabled', s => { s.repository.squashMergeAllowed = false; }, false],
  ['missing effective rules', s => { s.rules = []; }, false],
  ['force-push rule alone', s => { s.rules = [{ type: 'non_fast_forward' }]; }, false],
  ['non-strict checks', s => { s.rules[0].parameters.strict_required_status_checks_policy = false; }, false],
  ['missing required job', s => { s.rules[0].parameters.required_status_checks.pop(); }, false],
  ['check from another integration', s => { s.rules[0].parameters.required_status_checks[0].integration_id = 1; }, false],
  ['unbound check integration', s => { delete s.rules[0].parameters.required_status_checks[0].integration_id; }, false],
  ['draft PR', s => { s.repository.pullRequest.isDraft = true; }, false],
  ['closed PR', s => { s.repository.pullRequest.state = 'CLOSED'; }, false],
  ['wrong base branch', s => { s.repository.pullRequest.baseRefName = 'other'; }, false],
  ['human PR author', s => { s.repository.pullRequest.author.login = 'someone'; }, false],
  ['fork PR', s => { s.repository.pullRequest.headRepository.nameWithOwner = 'other/giano'; }, false],
  ['head changed after event', s => { s.repository.pullRequest.headRefOid = 'new-head'; }, false],
  ['no changed files', s => { s.files = []; }, false],
  ['source change', s => { s.files.push({ status: 'modified', filename: 'src/giano/model.py' }); }, false],
  ['added workflow', s => { s.files = [{ status: 'added', filename: '.github/workflows/new.yml' }]; }, false],
  ['deleted manifest', s => { s.files = [{ status: 'removed', filename: 'pyproject.toml' }]; }, false],
  ['renamed file', s => { s.files[0].status = 'renamed'; }, false],
];

test('Dependabot auto-merge accepts only verified eligible policy inputs', async t => {
  for (const [name, mutate, eligible] of cases) {
    await t.test(name, async () => {
      const state = baseline();
      mutate(state);
      const result = await run(state);
      assert.equal(result.outputs.eligible === 'true', eligible);
      if (eligible) {
        assert.deepEqual(result.failures, []);
        assert.equal(result.outputs['head-sha'], 'expected-head');
      }
    });
  }
});
