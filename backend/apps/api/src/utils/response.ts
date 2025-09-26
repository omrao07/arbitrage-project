// utils/response.ts
// Helpers to send JSON responses (pure Node, no imports)

function send(res, status, data, headers = {}) {
  const body = data === undefined ? "" : JSON.stringify(data);
  const h = {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body).toString(),
    ...headers,
  };
  res.writeHead(status, h);
  res.end(body);
}

export const ok = (res, data) => send(res, 200, data);
export const created = (res, data) => send(res, 201, data);
export const noContent = (res) => send(res, 204, "");
export const badRequest = (res, msg = "Bad Request") =>
  send(res, 400, { error: msg });
export const unauthorized = (res, msg = "Unauthorized") =>
  send(res, 401, { error: msg });
export const forbidden = (res, msg = "Forbidden") =>
  send(res, 403, { error: msg });
export const notFound = (res, msg = "Not Found") =>
  send(res, 404, { error: msg });
export const error = (res, msg = "Internal Error") =>
  send(res, 500, { error: msg });

export { send };