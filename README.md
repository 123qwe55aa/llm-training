# MIT 6.036 Course Reader

An eye-friendly local course reader built from MIT Open Learning Library's
6.036 Introduction to Machine Learning materials.

The site brings course notes, PDF content, video transcripts, exercises, labs,
and homework into one searchable reading interface.

## Run locally

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/week1-blog/index.html
```

## Rebuild the site manifest

```bash
python3 scripts/build_course_site.py
```

The repository contains question text and code templates, but does not include
exercise answers.
