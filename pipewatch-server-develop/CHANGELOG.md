# CHANGELOG

## v0.3.8 (2024-06-20)

### Fix

* fix: conversion error after lookup ([`8a4df71`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/8a4df712b5df753609690a327e51a6e3960de821))

## v0.3.7 (2024-06-20)

### Chore

* chore(release): v0.3.7 ([`bee23e5`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/bee23e55975b145f372b75f26a430a543821f852))

### Fix

* fix: use concurrent log handler for proper logging ([`6638632`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/66386329a87bfb4bf030b3a6b197fc7ecf8e5f9c))

* fix: increase lookup window for schedules ([`53f32c4`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/53f32c406a614abc45727e06964caea40b62d808))

## v0.3.6 (2024-06-19)

### Chore

* chore(release): v0.3.6 ([`c539b43`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/c539b43606b43f0b3017ecf1b79d472fbadf58dd))

### Ci

* ci: update package-tools ([`c64d44d`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/c64d44d322b2c5ee9d47ada96483e0700f45d7cf))

### Fix

* fix: fix job pipeline API lookup ([`8517f57`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/8517f572d8dd9e7040153072b4be094dc46a048d))

* fix: add missing job status ([`8878369`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/8878369c0269f9fb060a4447811eb54d8c0002f3))

## v0.3.5 (2024-03-01)

### Chore

* chore(release): v0.3.5 ([`acefc47`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/acefc47fb955f974dea47d54d4a0edb7df9d4a51))

* chore: update links after project transfer ([`7248ad9`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/7248ad97e9d6e3312c1f04a37bd7c239018e362f))

### Performance

* perf: improved error handling for template rendering logic ([`fa2c3da`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/fa2c3da3c2557c2d6d8e0b9b91c8e84882eb93c1))

## v0.3.4 (2024-02-15)

### Chore

* chore(release): v0.3.4 ([`0df8e59`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/0df8e593877660b2787d73a26e669ff3284c10c8))

### Fix

* fix: add missing dependency ([`7cb6f1a`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/7cb6f1a67edb60ae687643a9757ee224ed462fbd))

## v0.3.3 (2024-02-14)

### Chore

* chore(release): v0.3.3 ([`7098d0f`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/7098d0f6aaad80dfc9bd46f50ff1d9fdac8a3e3d))

### Performance

* perf: add more API callables to template context ([`c202de6`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/c202de67f74ad51eb84a29a818ca2cbc5684acdf))

## v0.3.2 (2024-02-01)

### Chore

* chore(release): v0.3.2 ([`1c50d04`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/1c50d046599411e9e3e395cf5203ee72d5535585))

### Fix

* fix: improve handling of multiple database spaces ([`8163c4d`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/8163c4db9a452aca71505bbdb8b16fe37082ccf2))

### Performance

* perf: allow table columns to be added and dropped ([`e69f73e`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/e69f73e15ca0eed6ac2085cb479efe9bfc98e26d))

### Refactor

* refactor: replace duplicate code with already available function ([`64f28fd`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/64f28fdb5d952da43cfa06ea71e0c754b0b763bd))

## v0.3.1 (2024-01-25)

### Chore

* chore(release): v0.3.1 ([`6b0c941`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/6b0c941110d68250e5a546f8034645c22d8eb716))

### Fix

* fix: solve issues with broadcasting during table merging ([`a0bdd47`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/a0bdd47c63298f8bef6ae3004c76dffebb61b8c5))

* fix: &#34;escape&#34; database queries to support special characters in table name ([`40a31fe`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/40a31fef7e633e1ad330ed502ba60d8b886dbc4d))

## v0.3.0 (2024-01-22)

### Chore

* chore(release): v0.3.0 ([`9c52038`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/9c52038b375fe6851265157052e01353cb6e2449))

### Documentation

* docs: fix badges links ([`8f89188`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/8f891887dc3216fa2c75cd824027930b9404071e))

### Feature

* feat: embed API objects into templating environment for table configurations ([`474df6e`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/474df6e6703eae8af86fb7d13ff309e84b50ecb4))

* feat: make all tables configurable ([`cc3229e`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/cc3229e376aa4f618eb8a6e92d157e1d77e1f1fd))

### Fix

* fix(logs): allow correct package-lib logging level ([`5bec3c8`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/5bec3c8bd2b17b4e429c1cbaa1609e712db67d29))

* fix: remove unnecessary connection to database ([`dd07c82`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/dd07c820c36007f7b4edfc92abc107ab9fe2b92d))

### Performance

* perf: remove unused webhook attributes from model and instead expose entire webhook to templating environment ([`99aefa7`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/99aefa7c50cdb683d4ff4bb749cbf5f0a5c0f777))

* perf: adapt core to changes in library model ([`f7ccd09`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/f7ccd09fc07379d8f39d8dcfc0bac9a03d957ff5))

* perf: allow different config file extensions ([`859912e`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/859912eb2120e581fcfab69916e01bfb02403a38))

* perf: check for overlapping table keys ([`6c58ab4`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/6c58ab4d8fdf5434907d013dc89d3cd8ce4fda2a))

* perf(logs): log test artifact ([`307577d`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/307577d1136f7670758b53d88ff1f673d5d9904a))

* perf: extend webhook models ([`4d9ea63`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/4d9ea630ab66eb4ba3feabfd67b9d76f6cc4c8d9))

### Refactor

* refactor: minor improvements ([`93ec6da`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/93ec6da8b852d0a326a6e4cab7c85ca7d63f4091))

## v0.2.0 (2024-01-17)

### Chore

* chore(release): v0.2.0 ([`041808c`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/041808cfded5075f416049f30a7d483bdff60e07))

* chore: add default log config file ([`c7ef36b`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/c7ef36bf57d0aa0cd5c2c5e8dc5d723c713df5e3))

### Ci

* ci: update package-tools version ([`436d3a5`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/436d3a50105d6e26320698981f8cd1b5615e4bca))

* ci: fix container publishing URL ([`4fe5202`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/4fe520237eab9ae4615ff648cd71e6c1c5cfe16b))

### Feature

* feat: implement endpoint for testing ([`bc09167`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/bc0916793e2f956c04c60a09eca61ecb7f51b9a6))

### Fix

* fix(db): change wrongly mapped SQL data type ([`8c25c5c`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/8c25c5c5ec79ca86bbd6017515c58bdaa75874e1))

* fix: solve issue where an error is raised when artifacts are not found

When fetching a specific artifact from a job which has artifacts, but not the specifically requested one, it seems that
the &#39;requests&#39; package raises a &#39;ChunkedEncodingError&#39; instead of a 404 or similar (see
https://github.com/python-gitlab/python-gitlab/issues/1186). This doesn&#39;t happen for jobs which do not have any
artifacts at all.
It is assumed that the error is safe to catch and ignore, which is done in this fix. ([`33db16d`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/33db16daf9c3196f64bb4f8f0b236e9fdd5f780f))

* fix: add missing status enums ([`89af0db`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/89af0db71c12a60eacd72394637de5e3eb51fcf8))

* fix: retry on chunked encoding errors ([`6a15966`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/6a159668d99409dd9af0ed9cbf76eb8a46911821))

### Performance

* perf: map string to SQL &#34;TEXT&#34; instead of &#34;VARCHAR&#34; ([`e9d6c1c`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/e9d6c1c040087b7d1377c1d9df41ae8c42603bc5))

* perf(lib): handle optional variables properly ([`f1f88c5`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/f1f88c553fa10b23c1b5df91504b26af2fbe6f7e))

* perf: improve endpoint configurability &amp; streamline configuration file ([`56eec5a`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/56eec5a75893b21ee7dfc8d4fef2c4685920fd82))

* perf: inject timestamps into tables ([`039b124`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/039b124413696111226259c17da2ffcdfaba351d))

* perf: enable authentication for webhooks ([`9bc1a25`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/9bc1a25c2e5878ebd29caa85ea6358186e193365))

### Refactor

* refactor: renaming database &#34;collections&#34; to &#34;spaces&#34; ([`3f90e08`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/3f90e08c1ea2ea8d797858863122ccdb879f0950))

* refactor: unify config and API endpoints ([`6c3e9ef`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/6c3e9efc4d5112e8bb29ded6cb1df91406335265))

* refactor: rename function for clarity ([`fbce5f1`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/fbce5f1970aaad49525d4c486dc442e1b7e12ff8))

* refactor: adapt to new package-lib features ([`0a46ce2`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/0a46ce28923f8c91bdea766150acbcf801ad0077))

## v0.1.1 (2023-12-11)

### Chore

* chore(release): v0.1.1 ([`fd81780`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/fd81780f134afc32c15b5ceda638837158fa421d))

### Fix

* fix(gitlab): validate schedule attributes dictionary instead of object ([`5afae38`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/5afae38677989135c96361b3836404e56c76ba93))

* fix: retrieve last page for reverse pipeline schedule lookup ([`67d8da8`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/67d8da87625a57c7ce0705e46c7258c2df53c2d3))

* fix: remove bad imports ([`5852046`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/5852046bcac2e236d4b0736e1b95a0ae454728ad))

* fix: change wrong container registry URL ([`8c5ac57`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/8c5ac5791e3149b49d5f77dd322e07f7c05e0b54))

### Performance

* perf: make passwords non-optional ([`b0b789f`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/b0b789fc1d72ee74067227e8b0206b0c946da43c))

## v0.1.0 (2023-12-08)

### Chore

* chore(release): v0.1.0 ([`5cd3f05`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/5cd3f05ace5f51dd219572a7b2c9ccfb29578e1b))

* chore: initial commit ([`6bd75c3`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/6bd75c3e14f0dba25f0521ae9a938f999e6ae32f))

### Documentation

* docs: remove docs ([`303d28b`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/303d28b38c1d498d52c1c565686a7e352b1e2f01))

### Feature

* feat: first version ([`367c115`](https://git.axelera.ai/ai-hw-team/pipewatch/pipewatch-server/-/commit/367c1152502c8bf89a6433485043469588df4c60))
