import path from 'path';

export function aliasPlugin(aliasMap) {
  const entries = Object.entries(aliasMap || {});
  if (!entries.length) return { name: 'arcane-alias', setup() {} };

  return {
    name: 'arcane-alias',
    setup(build) {
      build.onResolve({ filter: /.*/ }, (args) => {
        for (const [key, target] of entries) {
          if (args.path === key || args.path.startsWith(key + '/')) {
            const rest = args.path.slice(key.length);
            const full = path.join(target, rest);
            return { path: full };
          }
        }
        return null;
      });
    }
  };
}
