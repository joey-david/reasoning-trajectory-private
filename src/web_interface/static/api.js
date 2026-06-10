export async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

export async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

export async function watchJob(job, onUpdate) {
  onUpdate?.(job);
  while (job.status === "queued" || job.status === "running") {
    await sleep(700);
    job = await getJSON(`/api/jobs/${job.id}`);
    onUpdate?.(job);
  }
  if (job.status === "error") throw new Error(job.error);
  return job;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
