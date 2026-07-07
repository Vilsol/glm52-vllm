# LMCache v14 DCP patch set — vendored snapshot

These 12 patches are a **pinned snapshot** from the community repo
`github.com/myshytf/glm-5.2-v11-lmcache` (patches/), adapted for the GLM-5.2 v14
image. They are applied at container start by `../serve_glm52_v14_lmcache.sh`
against a pip-installed `lmcache==0.4.6`.

They are tightly coupled to lmcache 0.4.6 and the eldritch-v7 vLLM fork; expect
to refresh them when the community image updates or when festr bakes LMCache into
the official image. Retrieved 2026-07-07.
