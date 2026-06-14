const root = document.documentElement;
const body = document.body;
const themeButtons = document.querySelectorAll(".theme-toggle");

if (localStorage.getItem("week1-theme") === "night") body.classList.add("night");
themeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    body.classList.toggle("night");
    localStorage.setItem("week1-theme", body.classList.contains("night") ? "night" : "day");
  });
});

const escapeHtml = (value = "") => value
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

const inline = (value) => escapeHtml(value)
  .replace(/`([^`]+)`/g, (_, m) => {
    if (/^_+$/.test(m)) return `<code class="answer-blank">${m}</code>`;
    return `<code>${m}</code>`;
  })
  .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
  .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

const cleanContent = (markdown) => {
  let cleaned = markdown
    .replaceAll("No playable video sources found.", "")
    .replaceAll("Your browser does not support this video format. Try using a different browser.", "")
    .replaceAll("0:00 / 0:00", "");

  const lines = cleaned.replace(/\r/g, "").split("\n");
  let headingText = null;
  let headingRepeatBudget = 0;
  const filtered = lines.filter((line) => {
    const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      headingText = headingMatch[2].trim();
      headingRepeatBudget = 3;
      return true;
    }
    if (headingText && headingRepeatBudget > 0 && line.trim() === headingText) {
      headingRepeatBudget--;
      return false;
    }
    if (line.trim()) headingText = null;
    return true;
  });

  return filtered.join("\n")
    .replace(/\[mathjaxinline\](.*?)\[\/mathjaxinline\]/gs, (_, inner) => {
      inner = inner.trim();
      if (/^\$.+\$$/.test(inner)) return inner;
      return `$${inner}$`;
    })
    .replace(/\[mathjax\](.*?)\[\/mathjax\]/gs, (_, inner) => {
      inner = inner.trim();
      if (/^\$\$.+\$\$$/.test(inner)) return inner;
      return `$$${inner}$$`;
    })
    .replace(/\n{4,}/g, "\n\n\n");
};

const renderMarkdown = (markdown) => {
  const normalized = cleanContent(markdown);
  const lines = normalized.replace(/\r/g, "").split("\n");
  const html = [];
  let paragraph = [];
  let listType = "";
  let code = [];
  let inCode = false;
  let inTable = false;

  const closeParagraph = () => {
    if (paragraph.length) html.push(`<p>${inline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (listType) html.push(`</${listType}>`);
    listType = "";
  };
  const closeTable = () => {
    if (inTable) html.push("</tbody></table>");
    inTable = false;
  };
  const closeAll = () => { closeParagraph(); closeList(); closeTable(); };

  lines.forEach((line, index) => {
    if (line.startsWith("```")) {
      closeAll();
      if (inCode) {
        html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = [];
      }
      inCode = !inCode;
      return;
    }
    if (inCode) { code.push(line); return; }
    if (!line.trim()) { closeAll(); return; }
    if (/^---+$/.test(line.trim())) { closeAll(); html.push("<hr>"); return; }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      closeAll();
      const level = heading[1].length;
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      return;
    }
    if (line.startsWith("> ")) {
      closeAll();
      html.push(`<blockquote>${inline(line.slice(2))}</blockquote>`);
      return;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      closeParagraph();
      const wanted = bullet ? "ul" : "ol";
      if (listType !== wanted) { closeList(); listType = wanted; html.push(`<${wanted}>`); }
      html.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      return;
    }
    if (line.includes("|") && lines[index + 1]?.match(/^\|?[\s:-]+\|/)) {
      closeAll();
      const headers = line.split("|").filter(Boolean).map((cell) => `<th>${inline(cell.trim())}</th>`).join("");
      html.push(`<table><thead><tr>${headers}</tr></thead><tbody>`);
      inTable = true;
      return;
    }
    if (/^\|?[\s:-]+\|/.test(line)) return;
    if (line.includes("|") && inTable) {
      const cells = line.split("|").filter(Boolean).map((cell) => `<td>${inline(cell.trim())}</td>`).join("");
      html.push(`<tr>${cells}</tr>`);
      return;
    }
    // Paragraph break: flush if line looks like start of new paragraph
    const pLine = line.trim();
    if (paragraph.length > 0) {
      const lastLine = paragraph[paragraph.length - 1];
      const endsSentence = /[.?!:]$/.test(lastLine);
      const startsNew = /^[A-Z"'(]/.test(pLine) || /^\d+\.\d/.test(pLine);
      if ((endsSentence && startsNew) || /^\d+\.\d+/.test(pLine)) {
        closeParagraph();
      }
    }
    paragraph.push(pLine);
  });
  closeAll();
  if (inCode) html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  return html.join("\n");
};

// --- Unicode box-drawing matrix to LaTeX bmatrix converter ---
const matrixBracketOpen = /[\u23A1\u23A3]/;
const matrixBracketClose = /[\u23A4\u23A6]/;
const matrixDecor = /^[\u23A0-\u23B3\u2500\u2502]$/;

const detectMatrixBlock = (lines, startIdx) => {
  const firstLine = lines[startIdx].trim();
  if (!matrixBracketOpen.test(firstLine)) return null;

  let endIdx = -1;
  for (let i = startIdx; i < lines.length; i++) {
    if (matrixBracketClose.test(lines[i].trim())) {
      endIdx = i;
      break;
    }
  }
  if (endIdx === -1) return null;

  const bodyLines = [];
  let contentStarted = false;
  let labelBefore = "";
  for (let i = startIdx + 1; i < endIdx; i++) {
    const trimmed = lines[i].trim();
    if (matrixDecor.test(trimmed)) continue;
    if (!trimmed) continue;
    const cleanLine = trimmed.replace(/[\t\u200B]+/g, "").trim();
    if (cleanLine) bodyLines.push(cleanLine);
  }
  if (bodyLines.length === 0) return null;

  for (let i = startIdx - 1; i >= Math.max(0, startIdx - 3); i--) {
    const before = lines[i].trim();
    const labelMatch = before.match(/^([a-zA-Z_]\w*)\s*=\s*$/);
    if (labelMatch) {
      labelBefore = labelMatch[1];
      break;
    }
  }
  return { startIdx, endIdx, label: labelBefore, rows: bodyLines };
};

const matrixToLatex = (block) => {
  let latex = "";
  if (block.label) latex = `${block.label}=`;
  latex += "\\begin{bmatrix}\n";
  latex += block.rows.map((row) => row.replace(/⋮/g, "\\vdots")).join(" \\\\\n");
  latex += "\n\\end{bmatrix}";
  return `$$${latex}$$`;
};

const activityToMarkdown = (text) => {
  let cleaned = text
    .replaceAll("Submit View Answer Ask for Help", "")
    .replaceAll("Run Code Submit View Answer Ask for Help", "")
    .replaceAll("Check Syntax Submit View Answer Ask for Help", "")
    .replaceAll("Submit View Answer", "")
    .replaceAll("You have infinitely many submissions remaining.", "");
  cleaned = cleaned.replace(/[\u200B\t\r]+/g, "");
  cleaned = cleaned.replace(/\n{4,}/g, "\n\n\n");

  const lines = cleaned.split("\n");

  // Phase 1: Detect and convert box-drawing matrix blocks to LaTeX
  const converted = [];
  let i = 0;
  while (i < lines.length) {
    const block = detectMatrixBlock(lines, i);
    if (block) {
      converted.push(matrixToLatex(block));
      i = block.endIdx + 1;
    } else {
      converted.push(lines[i]);
      i++;
    }
  }

  const cleanLines = [];
  const mathSurrogate = /^[\uD800-\uDBFF][\uDC00-\uDFFF]$/;
  const bmpMathChar = /^[\u2100-\u214F\u2032\u23A0-\u23B3\u22EE\u22F0-\u22F1]$/;
  const singleChar = /^[\uD800-\uDBFF][\uDC00-\uDFFF]|^[=(\\)),;:\d\-−′∙⋅+*\/<>^]$/;
  let inMathBlock = false;

  for (const raw of converted) {
    const trimmed = raw.trim();
    if (!trimmed) { inMathBlock = false; cleanLines.push(raw); continue; }
    if (mathSurrogate.test(trimmed) || bmpMathChar.test(trimmed)) { inMathBlock = true; continue; }
    if (inMathBlock) {
      if (singleChar.test(trimmed)) continue;
      inMathBlock = false;
    }
    cleanLines.push(raw);
  }

  const out = [];
  cleanLines.forEach((line) => {
    const val = line.trimEnd();
    if (/^\d+\)\s+\S/.test(val)) { out.push(`## ${val}`); return; }
    if (/^\d+\.\d+\)\s+\S/.test(val)) { out.push(`### ${val}`); return; }
    if (/^\d+\.\d+[a-z]\)\s+\S/i.test(val)) { out.push(`#### ${val}`); return; }
    if (/^Ex\d/i.test(val)) { out.push(`### ${val}`); return; }
    out.push(val);
  });

  let result = out.join("\n");
  result = result.replace(/\( [^)]+? \)(?= ?\([^)]+?\))/g, "");

  // Merge orphan closing brackets into previous line
  const merged = [];
  const lines3 = result.split("\n");
  for (let i = 0; i < lines3.length; i++) {
    const curr = lines3[i];
    const trimmed = curr.trim();
    if (merged.length > 0 && (
      /^[)\]}\.,;:]/.test(trimmed) ||
      (trimmed.length <= 3 && /^[)\]}\.,;:\d]/.test(trimmed))
    )) {
      merged[merged.length - 1] += curr;
    } else {
      merged.push(curr);
    }
  }
  result = merged.join("\n");

  // Merge separated letter-digit-pairs across lines
  result = result.replace(/([\u00C0-\u024F\u0370-\u03FF\u210E]|[A-Z])[ \n](\d)/g, "$1$2");
  result = result.replace(/\b([A-Z])\n([a-z])(?=[^a-z]|$)/g, "$1$2");
  result = result.replace(/(\w)\n{2,}(?=[(\[⟨])/g, "$1 ");
  result = result.replace(/(\w)\n+\)/g, "$1)");
  result = result.replace(/\(\s*\(/g, "(");
  result = result.replace(/\)\s*\)/g, ")");
  result = result.replace(/^ +$/gm, "");
  result = result.replace(/\n{3,}/g, "\n\n");

  // Wrap standalone display math
  const lines2 = result.split("\n");
  const mathWrapped = [];
  for (let i = 0; i < lines2.length; i++) {
    const raw = lines2[i];
    const trimmed = raw.trim();
    const isHeading = /^#/.test(trimmed) || /^\d+[).]/.test(trimmed) || /^Ex\d/i.test(trimmed);
    const startsWithWord = /^(Consider|For|Which|What|Enter|Select|Does|There|As|The|Provide|If|This|These|In|Note|That|You|Select|Such|All|We|It|A)/.test(trimmed);
    const isURL = trimmed.startsWith("http") || trimmed.startsWith("ftp");
    const isShort = trimmed.length > 2 && trimmed.length < 50 && trimmed.split(/\s+/).length <= 4;
    const hasEquals = /=/.test(trimmed);
    const startsMath = /^[A-Za-zθ𝜃𝜙𝜓𝛽αδϵεπμλωΩΣΔΠΓℰℒℋ𝒙𝒚𝒛𝑤𝐸ℎ'′′″]/.test(trimmed);
    if (!isHeading && !startsWithWord && !isURL && isShort && hasEquals && startsMath) {
      mathWrapped.push(`$$${trimmed}$$`);
    } else {
      mathWrapped.push(raw);
    }
  }
  result = mathWrapped.join("\n");

  // === Inline math wrapping for common patterns ===
  // Vector/matrix bracket notation: "[1 5 -3 2]" -> "$[1\\ 5\\ -3\\ 2]$"
  result = result.replace(/(?<!\$)\[(\d[\d\s\-.,]*\d)\](?!\w)/g, (m) => {
    const inner = m.slice(1,-1).trim().replace(/\s+/g, '\\ ');
    return `$$\\begin{bmatrix}${inner}\\end{bmatrix}$$`;
  });
  // Transpose: "[1 5 -3 2]T" or "[1 5 -3 2]^T"
  result = result.replace(/(?<!\$)\[(\d[\d\s\-.,]*\d)\]\s*[Tt]/g, (m) => {
    const inner = m.slice(1,-1).replace(/[Tt]$/,'').trim().replace(/\s+/g, '\\ ');
    return `$$\\begin{bmatrix}${inner}\\end{bmatrix}^T$$`;
  });
  // Subscripts: "c_0", "c_i", "c_{n-1}" etc
  result = result.replace(/\b([A-Za-zθ𝜙ψβαδϵεπμλω])_(?:\{([^}]+)\}|(\d))/g, '$1_{$2$3}');
  // Superscripts: "c^(1)" -> "c^{(1)}", "a^T" -> "a^{T}"
  result = result.replace(/\b([A-Za-z])\^\((\d+)\)/g, '$1^{($2)}');
  result = result.replace(/\b([A-Za-zθ𝜙ψβαδϵ])\^T/g, '$1^{T}');
  // Greek letters as standalone words
  result = result.replace(/\btimes\b/g, '\\times');
  result = result.replace(/\bcdot\b/g, '\\cdot');
  result = result.replace(/\bdot\b/g, '\\cdot');
  // Multiplication dot: "a⋅b" -> "a \\cdot b" (within inline math context)
  // Simple inline math: standalone expressions like "a+b", "n=4", "a⋅b"
  result = result.replace(
    /(?<!\$)(^|[^A-Za-z0-9])([A-Za-zθ𝜙ψβαδϵεπμλω]\s*[+\-−∗⋅×/=]\s*[A-Za-zθ𝜙ψβαδϵεπμλω0-9][A-Za-z0-9+\-−∗⋅×=/()]*)(?=[.,;:!?\s]|$)/gm,
    (m, before, expr) => `${before}\\$${expr.trim()}\\$`
  );

  // Mark yes/no and input locations
  result = result.replace(/\?\s*\n--\s*\n(yes|no)\s*\n(yes|no)/g, '? {RADIO:$1,$2}');
  result = result.replace(/(does\?)\s*--\s*(yes)\s*(no)/gi, '? {RADIO:$2,$3}');
  result = result.replace(/(exists\?)\s*--\s*(yes)\s*(no)/gi, '? {RADIO:$2,$3}');
  result = result.replace(/\? -- (yes|no)(?:\s+(yes|no))?/g, '? {RADIO:$1,$2}');
  result = result.replace(/-- (yes|no)\s+(yes|no)/g, '{RADIO:$1,$2}');
  result = result.replace(/(Enter a Python list[^:]*:)/g, '$1 {INPUT}');

  return result;
};

const curatedSources = {
  lecture: {
    title: "从数据到线性分类器",
    kicker: "WEEK 1 · 中文精编讲义",
    deck: "九段课程视频的中文知识骨架：从泛化问题出发，走到第一个可执行的学习算法。",
    url: "../downloads/MITx-6.036-Week1-lecture-notes/Week1-%E8%A7%86%E9%A2%91%E8%AE%B2%E4%B9%89.md",
    markdown: true,
  },
  exercises: {
    title: "Hyperplanes 与 NumPy",
    kicker: "WEEK 1 · 原版练习册（清理版）",
    deck: "原始题目与代码模板，已清理为可读 Markdown。可线下用 Python 练习。",
    url: "../downloads/MITx-6.036-Week1-exercises/week1_exercises.md",
    markdown: true,
  },
  introduction: {
    title: "Introduction to Machine Learning",
    kicker: "课程原典 · CHAPTER 1",
    deck: "机器学习的问题版图：从 induction、estimation 与 generalization，走向模型类与算法。",
    url: "content/introduction.md",
    markdown: true,
  },
  linear: {
    title: "Linear Classifiers",
    kicker: "课程原典 · CHAPTER 2",
    deck: "第一种具体的假设类，以及如何学习、评价一个线性分类器。",
    url: "content/linear-classifiers.md",
    markdown: true,
  },
};

const coursePromise = fetch("content/course.json").then((response) => {
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}).then((course) => {
  const articles = {};
  course.chapters.forEach((chapter) => {
    chapter.sections.forEach((section) => {
      const hasLti = section.lti_blocks && section.lti_blocks.length > 0;
      if (hasLti) {
        if (section.article) {
          articles[section.id] = {
            title: section.title,
            kicker: `${chapter.title} · ${typeLabels[section.type] || "讲义"}`,
            deck: `${section.video_count} 段视频 · ${section.pdf_count ?? section.pdf_urls?.length ?? 0} 份 PDF`,
            url: section.article,
            markdown: true,
          };
        } else {
          const blockName = section.lti_blocks[0];
          articles[section.id] = {
            title: section.title,
            kicker: `${chapter.title} · ${typeLabels[section.type] || "讲义"}`,
            deck: `${section.characters} 字符`,
            url: `content/lti/${blockName}.txt`,
            markdown: false,
          };
        }
        section.lti_blocks.forEach((block) => {
          const blockArticle = section.article
            ? section.article.replace(/\/\w+\.md$/, `/${block}.md`)
            : null;
          articles[`activity-${block}`] = {
            title: `${section.title} (${block})`,
            kicker: `${chapter.title} · Exercise`,
            deck: blockArticle ? "已清理" : "",
            url: blockArticle || `content/lti/${block}.txt`,
            markdown: !!blockArticle,
          };
        });
      } else if (section.article) {
        articles[section.id] = {
          title: section.title,
          kicker: `${chapter.title} · ${typeLabels[section.type] || "讲义"}`,
          deck: `${section.video_count} 段视频 · ${(section.pdf_count ?? section.pdf_urls?.length ?? 0)} 份 PDF`,
          url: section.article,
          markdown: true,
        };
      }
    });
  });
  course.articles = articles;
  return course;
});

const typeLabels = {
  notes: "讲义",
  overview: "活动导读",
  exercise: "Exercise",
  lab: "Lab",
  homework: "Homework",
  guide: "课程信息",
};

const renderCatalog = (course) => {
  const catalog = document.querySelector("#course-catalog");
  if (!catalog) return;
  if (course.stats) {
    Object.entries(course.stats).forEach(([key, value]) => {
      const node = document.querySelector(`#stat-${key}`);
      if (node) node.textContent = String(value).padStart(2, "0");
    });
  }

  const chapterHtml = course.chapters.map((chapter, chIdx) => {
    const chNum = chapter.number ?? (chIdx + 1);
    return `
    <details class="course-chapter" ${chNum <= 1 ? "open" : ""}>
      <summary>
        <span class="chapter-number">${String(chNum).padStart(2, "0")}</span>
        <span><strong>${escapeHtml(chapter.title)}</strong><small>${chapter.sections.length} 个章节</small></span>
        <span class="chapter-arrow">展开</span>
      </summary>
      <div class="chapter-sections">
        ${chapter.sections.map((section) => {
          const pdfCount = section.pdf_count ?? (section.pdf_urls?.length ?? 0);
          const secType = section.type || "notes";
          return `
          <article class="course-section" data-search="${escapeHtml(`${chapter.title} ${section.title} ${(section.activities||[]).map((item) => item.title).join(" ")}`.toLowerCase())}">
            <a class="section-main" href="reader.html?article=${encodeURIComponent(section.id)}">
              <span class="type-badge ${secType}">${typeLabels[secType]}</span>
              <span class="section-copy">
                <strong>${escapeHtml(section.title)}</strong>
                <small>${section.video_count} 段视频 transcript · ${pdfCount} 份 PDF 已融合</small>
              </span>
              <span class="section-open">阅读 →</span>
            </a>
            ${(section.lti_blocks||[]).length ? `<div class="activity-links">${section.lti_blocks.map((block) => `
              <a href="reader.html?article=${encodeURIComponent("activity-" + block)}">
                <span class="type-badge exercise">Exercise</span>
                ${escapeHtml(block.replace(/_/g, " "))}
              </a>`).join("")}</div>` : ""}
          </article>`;
        }).join("")}
      </div>
    </details>`;}).join("");
  catalog.innerHTML = chapterHtml;

  const search = document.querySelector("#course-search");
  search?.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();
    document.querySelectorAll(".course-section").forEach((section) => {
      section.hidden = query && !section.dataset.search.includes(query);
    });
    document.querySelectorAll(".course-chapter").forEach((chapter) => {
      const matches = [...chapter.querySelectorAll(".course-section")].some((section) => !section.hidden);
      chapter.hidden = !matches;
      if (query && matches) chapter.open = true;
    });
  });
};

const articleContent = document.querySelector("#article-content");
if (articleContent) {
  const params = new URLSearchParams(location.search);
  const requested = params.get("article") || "intro_ml";
  coursePromise
    .then((course) => curatedSources[requested] || course.articles[requested] || course.articles["intro_ml"])
    .then((article) => {
      if (!article) throw new Error("article not found");
      document.title = `${article.title} · MIT 6.036`;
      document.querySelector("#article-title").textContent = article.title;
      document.querySelector("#article-kicker").textContent = article.kicker;
      document.querySelector("#article-deck").textContent = article.deck;
      return fetch(article.url).then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      }).then((text) => ({ article, text }));
    })
    .then(({ article, text }) => {
      const rendered = article.markdown
        ? renderMarkdown(text)
        : renderMarkdown(activityToMarkdown(text));
      if (!article.markdown) articleContent.classList.add("activity-view");
      articleContent.innerHTML = rendered;
      articleContent.querySelectorAll("p").forEach((p) => {
        p.innerHTML = p.innerHTML.replace(
          /\{RADIO:(\w+),(\w+)\}/g,
          (_, v1, v2) => {
            const name = "ex-" + Math.random().toString(36).slice(2, 6);
            return '<div class="exercise-options">'
              + '<label class="ex-option"><input type="radio" name="' + name + '" value="' + v1 + '"> ' + v1 + '</label>'
              + '<label class="ex-option"><input type="radio" name="' + name + '" value="' + v2 + '"> ' + v2 + '</label>'
              + '</div>'
              + '<div class="ex-buttons">'
              + '<button class="ex-btn-submit" disabled>Submit</button>'
              + '<button class="ex-btn-view" disabled>View Answer</button>'
              + '</div>'
              + '<div class="ex-submissions">You have infinitely many submissions remaining.</div>';
          }
        );
        p.innerHTML = p.innerHTML.replace(
          /\{INPUT\}/g,
          '<div class="exercise-input"><input type="text" class="ex-text-input" placeholder="Type your answer here..."></div>'
          + '<div class="ex-buttons">'
          + '<button class="ex-btn-submit" disabled>Submit</button>'
          + '<button class="ex-btn-view" disabled>View Answer</button>'
          + '</div>'
          + '<div class="ex-submissions">You have infinitely many submissions remaining.</div>'
        );
      });
      const titleHeader = articleContent.querySelector("h1");
      if (titleHeader && articleContent.classList.contains("activity-view")) {
        const points = document.createElement("div");
        points.className = "ex-points";
        points.textContent = "(1.0 points possible)";
        titleHeader.after(points);
      }
      const firstP = articleContent.querySelector("p");
      if (firstP) {
        firstP.querySelectorAll("a").forEach((a) => a.classList.add("ex-hyperlinks"));
      }
      const toc = document.querySelector("#toc");
      [...articleContent.querySelectorAll("h2, h3")].forEach((heading, index) => {
        heading.id = `section-${index + 1}`;
        const link = document.createElement("a");
        link.href = `#${heading.id}`;
        // Strip $...$ from TOC text - KaTeX markers shouldn't appear in nav
        link.textContent = heading.textContent.replace(/\$[^$]*\$/g, '').replace(/\s+/g, ' ').trim();
        if (heading.tagName === "H3") link.style.paddingLeft = "24px";
        toc.appendChild(link);
      });
      if (window.renderMathInElement) {
        renderMathInElement(articleContent, {
          delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
          ],
          throwOnError: false
        });
      }
      // Restore user highlights after content is fully rendered
      applyHighlights(requested);
    })
    .catch(() => {
      articleContent.innerHTML = '<div class="loading-error">内容读取失败。请确认本地课程资料与服务器均已启动。</div>';
    });

  document.querySelectorAll("[data-font]").forEach((button) => {
    button.addEventListener("click", () => {
      const current = parseInt(getComputedStyle(root).getPropertyValue("--reading-size"), 10) || 18;
      const next = button.dataset.font === "up" ? Math.min(current + 1, 23) : Math.max(current - 1, 15);
      root.style.setProperty("--reading-size", `${next}px`);
    });
  });

  const progress = document.querySelector(".reading-progress span");
  addEventListener("scroll", () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = `${max > 0 ? (scrollY / max) * 100 : 0}%`;
  }, { passive: true });

  /* ── Highlighter (sentence-level) ── */
  const hlToggle = document.getElementById("highlight-toggle");
  const HIGHLIGHT_KEY = "llm-training:highlights";
  const HL_MODE_KEY = "llm-training:highlight-mode";

  function getHighlights() {
    try { return JSON.parse(localStorage.getItem(HIGHLIGHT_KEY)) || {}; } catch { return {}; }
  }

  function saveRangeHighlight(articleId, paraIndex, start, end, text) {
    const h = getHighlights();
    if (!h[articleId]) h[articleId] = {};
    if (!h[articleId][paraIndex]) h[articleId][paraIndex] = [];
    // Remove duplicate (same text range)
    h[articleId][paraIndex] = h[articleId][paraIndex].filter(
      r => !(r.start === start && r.end === end)
    );
    if (text) h[articleId][paraIndex].push({ start, end, text });
    if (h[articleId][paraIndex].length === 0) delete h[articleId][paraIndex];
    localStorage.setItem(HIGHLIGHT_KEY, JSON.stringify(h));
  }

  /** Wrap text nodes in a container with <mark> for a single character range */
  function applyRange(container, startOff, endOff) {
    const textNodes = [];
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
    let node;
    while ((node = walker.nextNode())) textNodes.push(node);

    let acc = 0;
    const toWrap = [];
    for (const tn of textNodes) {
      const len = tn.textContent.length;
      const segStart = Math.max(0, startOff - acc);
      const segEnd = Math.min(len, endOff - acc);
      if (segStart < segEnd) toWrap.push({ node: tn, start: segStart, end: segEnd });
      acc += len;
      if (acc >= endOff) break;
    }

    for (const { node, start, end } of toWrap) {
      const parent = node.parentNode;
      if (start === 0 && end === node.textContent.length && parent.tagName !== "MARK") {
        const mark = document.createElement("mark");
        mark.className = "hl";
        parent.replaceChild(mark, node);
        mark.appendChild(node);
      } else {
        const before = node.textContent.slice(0, start);
        const mid = node.textContent.slice(start, end);
        const after = node.textContent.slice(end);
        const frag = document.createDocumentFragment();
        if (before) frag.appendChild(document.createTextNode(before));
        const mark = document.createElement("mark");
        mark.className = "hl";
        mark.textContent = mid;
        frag.appendChild(mark);
        if (after) frag.appendChild(document.createTextNode(after));
        parent.replaceChild(frag, node);
      }
    }
  }

  function applyHighlights(articleId) {
    const h = getHighlights();
    const ranges = h[articleId] || {};
    // Remove existing <mark.hl> that we created
    articleContent.querySelectorAll("mark.hl").forEach(el => {
      const parent = el.parentNode;
      parent.replaceChild(document.createTextNode(el.textContent), el);
      parent.normalize();
    });
    // Re-apply from stored ranges
    for (const [idxStr, entries] of Object.entries(ranges)) {
      const el = articleContent.children[parseInt(idxStr)];
      if (!el) continue;
      for (const { start, end, text } of entries) {
        // Use stored offsets; if offsets don't match, search by text
        const fullText = el.textContent;
        if (start >= 0 && end <= fullText.length && fullText.slice(start, end) === text) {
          applyRange(el, start, end);
        } else {
          // Fallback: search text position
          const pos = fullText.indexOf(text);
          if (pos >= 0) applyRange(el, pos, pos + text.length);
        }
      }
    }
  }

  let hlActive = (() => {
    try { return localStorage.getItem(HL_MODE_KEY) === "true"; } catch { return false; }
  })();
  let hlFloat = null;

  if (hlToggle) {
    // Apply initial state
    document.body.classList.toggle("highlight-mode", hlActive);
    hlToggle.classList.toggle("active", hlActive);

    hlToggle.addEventListener("click", () => {
      hlActive = !hlActive;
      document.body.classList.toggle("highlight-mode", hlActive);
      hlToggle.classList.toggle("active", hlActive);
      try { localStorage.setItem(HL_MODE_KEY, hlActive); } catch {}
      if (!hlActive && hlFloat) { hlFloat.remove(); hlFloat = null; }
    });

    articleContent.addEventListener("mouseup", (e) => {
      if (!hlActive) return;
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.toString().trim()) {
        if (hlFloat) { hlFloat.remove(); hlFloat = null; }
        return;
      }
      // Show floating highlight button
      if (!hlFloat) {
        hlFloat = document.createElement("button");
        hlFloat.textContent = "🖍 高亮";
        hlFloat.style.cssText = "position:fixed;z-index:999;padding:4px 10px;border:1px solid var(--amber);background:var(--paper);color:var(--ink);border-radius:4px;cursor:pointer;font:14px var(--sans);box-shadow:0 2px 8px rgba(0,0,0,.2);";
        hlFloat.addEventListener("click", () => {
          const sel2 = window.getSelection();
          if (!sel2 || sel2.isCollapsed) return;
          const range = sel2.getRangeAt(0);
          const container = range.commonAncestorContainer;
          let para = container;
          while (para && para.parentElement !== articleContent) para = para.parentElement;
          if (!para || para === articleContent) return;
          const idx = Array.from(articleContent.children).indexOf(para);
          if (idx < 0) return;
          const text = sel2.toString().trim();
          // Calculate character offset within the paragraph
          const preRange = document.createRange();
          preRange.selectNodeContents(para);
          preRange.setEnd(range.startContainer, range.startOffset);
          const start = preRange.toString().length;
          const end = start + text.length;
          applyRange(para, start, end);
          saveRangeHighlight(requested, idx, start, end, text);
          sel2.removeAllRanges();
          hlFloat.remove(); hlFloat = null;
        });
        document.body.appendChild(hlFloat);
      }
      const rect = sel.getRangeAt(0).getBoundingClientRect();
      hlFloat.style.left = Math.min(rect.left + rect.width / 2 - 40, window.innerWidth - 120) + "px";
      hlFloat.style.top = (rect.bottom + 6) + "px";
    });

    document.addEventListener("mousedown", (e) => {
      if (hlFloat && e.target !== hlFloat && !hlFloat.contains(e.target)) {
        hlFloat.remove(); hlFloat = null;
      }
    });

    // Apply on load — after content is rendered and KaTeX is done
    // (moved inside the .then() chain, see above)
  }
} else {
  coursePromise.then(renderCatalog).catch(() => {
    const catalog = document.querySelector("#course-catalog");
    if (catalog) catalog.innerHTML = '<p class="loading-error">课程目录读取失败。</p>';
  });
}
