// Cloudflare Worker entry: /api/* goes to the API, everything else to static assets.

import { handleApi } from './api.js';
import { createStore, d1Adapter } from './store.js';

let store;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/')) {
      store ??= createStore(d1Adapter(env.DB));
      return handleApi(request, store);
    }
    return env.ASSETS.fetch(request);
  },
};
