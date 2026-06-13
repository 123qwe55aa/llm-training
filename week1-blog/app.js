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
  // 1. Strip video placeholder garbage
  let cleaned = markdown
    .replaceAll("No playable video sources found.", "")
    .replaceAll("Your browser does not support this video format. Try using a different browser.", "")
    .replaceAll("0:00 / 0:00", "");

  // 2. Strip duplicate heading text: heading text repeated verbatim up to 3 times after heading
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

  // 3. Convert mathjax tags to proper LaTeX delimiters
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
    // 4. Clean up excessive blank lines (collapse 3+ to 2)
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
    paragraph.push(line.trim());
  });
  closeAll();
  if (inCode) html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  return html.join("\n");
};

const activityToMarkdown = (text) => {
  // Strip UI interaction elements
  let cleaned = text
    .replaceAll("Submit View Answer Ask for Help", "")
    .replaceAll("Run Code Submit View Answer Ask for Help", "")
    .replaceAll("Check Syntax Submit View Answer Ask for Help", "")
    .replaceAll("Submit View Answer", "")
    .replaceAll("You have infinitely many submissions remaining.", "");
  // Strip zero-width spaces, tabs, and carriage returns
  cleaned = cleaned.replace(/[\u200B\t\r]+/g, " ");
  // Collapse excessive blank lines
  cleaned = cleaned.replace(/\n{4,}/g, "\n\n\n");

  const lines = cleaned.split("\n");
  const cleanLines = [];
  // Detect and remove broken Unicode math (each char on its own line)
  const mathSurrogate = /^[\uD800-\uDBFF][\uDC00-\uDFFF]$/;
  const bmpMathChar = /^[\u2100-\u214F\u2032]$/;
  const singleChar = /^[\uD800-\uDBFF][\uDC00-\uDFFF]|^[=(\)),;:\d\-−′∙⋅+*\/<>^]$/;
  let inMathBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const trimmed = raw.trim();
    if (!trimmed) {
      inMathBlock = false;
      cleanLines.push(raw);
      continue;
    }
    if (mathSurrogate.test(trimmed) || bmpMathChar.test(trimmed)) {
      inMathBlock = true;
      continue;
    }
    if (inMathBlock) {
      if (singleChar.test(trimmed)) {
        continue;
      }
      inMathBlock = false;
      cleanLines.push(raw);
      continue;
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
  // Remove spaced-out duplicate tuples
  result = result.replace(/\( [^)]+? \)(?= ?\([^)]+?\))/g, "");

  // === Post-processing: wrap math in $...$ for MathJax ===
  const lines2 = result.split("\n");
  const mathWrapped = [];

  for (let i = 0; i < lines2.length; i++) {
    const raw = lines2[i];
    const trimmed = raw.trim();

    // Detect display math: standalone equation line (θ=..., h(x;θ)=..., E_n(...)=...)
    const displayMathRE = /^[A-Za-zθ𝜃𝜙𝜓𝛽αδϵεπμλωΩΣΔΠΓℰℒℋ]*['′]?\s*[=(]\s*.*[)=\d]$/;
    // But not if it's already a heading, list, or looks like regular text
    const isHeading = /^#/.test(trimmed) || /^\d+[).]/.test(trimmed) || /^Ex\d/i.test(trimmed);
    const hasVerb = /^(Consider|For|Which|What|Enter|Select|Does|There|As|The|Provide|If)/.test(trimmed);
    const isURL = trimmed.startsWith("http") || trimmed.startsWith("ftp");

    if (!isHeading && !hasVerb && !isURL && trimmed.length > 5 && displayMathRE.test(trimmed)) {
      // Display math — wrap in $$...$$
      mathWrapped.push(`$$${trimmed}$$`);
      continue;
    }

    // For non-heading lines, wrap inline math expressions in $...$
    // Pattern: Greek letters and function names followed by parenthesized args
    let processed = raw;
    if (!isHeading && !trimmed.startsWith("$$")) {
      // Wrap common inline math patterns:
      // 1. Greek letters (θ, θ', ℰ(θ,θ₀), ℋ, 𝒙, etc.)
      // 2. Function calls like h(x;θ), E_n(θ,θ₀), En(θ,θ0)
      // 3. Math symbols like ℎ, 𝑛, etc.
      // But NOT numbers, list markers, or text like "yes no"
      
      // Inline math: function calls with Greek args
      processed = processed.replace(
        /\b([A-Za-z])(?:_\{?(\w+)\}?)?\s*\(([^)]*[θ𝜃𝜙𝜓𝛽αδϵεπμλωΩΣΔΠΓℰℒ][^)]*)\)/g,
        (_, fn, sub, args) => {
          const full = sub ? `${fn}_{${sub}}(${args})` : `${fn}(${args})`;
          return `$${full}$`;
        }
      );
      // Inline math: standalone Greek letters (not preceded by word chars)
      processed = processed.replace(/(?<!\w)([θ𝜃𝜙𝜓𝛽αδϵεπμλωΩΣΔΠΓℰℋ𝒙𝒚𝒛𝑤𝐸ℎ′]'?)(?!\w)/g, "$$$1$$");
      // Inline math: expressions that are mostly math like θ=(1,−1,2,−3) within text
      processed = processed.replace(/\b([θ𝜃𝜙𝜓𝛽αδϵεπμλωΩΣΔΠΓℰ]'?\s*=\s*\([^)]+\))\b/g, "$$$1$$");
      // Inline math: h(x;θ)=+1, h(x;θ), etc. (function name + parenthesized math args)
      processed = processed.replace(/\b([a-z])\([^)]*[θ𝜃𝜙𝜓β;=+\-−][^)]*\)/g, (m) => `$${m}$`);
    }

    mathWrapped.push(processed);
  }

  result = mathWrapped.join("\n");

  // === Convert -- yes/no to radio buttons ===
  // Pattern: "Does ...? -- yes no" → radio buttons
  result = result.replace(
    /\? -- (yes|no)(?:\s+(yes|no))?/g,
    (match, first, second) => {
      if (second) {
        // Two options: -- yes no
        return `?\n<div class="exercise-options">\n  <label class="ex-option"><input type="radio" name="ex-${Math.random().toString(36).slice(2, 6)}" value="${first}"> ${first}</label>\n  <label class="ex-option"><input type="radio" name="ex-${Math.random().toString(36).slice(2, 6)}" value="${second}"> ${second}</label>\n</div>`;
      }
      return match;
    }
  );

  // === Convert "Enter a Python list..." to text input ===
  result = result.replace(
    /(Enter a Python list[^:]*:)/g,
    '$1\n<div class="exercise-input"><input type="text" class="ex-text-input" placeholder="Type your answer here..."></div>'
  );

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
  // Build flat articles map from chapters/sections/activities
  const articles = {};
  course.chapters.forEach((chapter) => {
    chapter.sections.forEach((section) => {
      articles[section.id] = {
        title: section.title,
        kicker: `${chapter.title} · ${typeLabels[section.type] || "讲义"}`,
        deck: `${section.video_count} 段视频 · ${section.pdf_count} 份 PDF`,
        url: section.url,
        markdown: section.markdown !== false,
      };
      section.activities.forEach((activity) => {
        articles[activity.id] = {
          title: activity.title,
          kicker: `${chapter.title} · ${typeLabels[activity.type] || "练习"}`,
          deck: `${activity.characters} 字符`,
          url: activity.url,
          markdown: false,
        };
      });
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
  Object.entries(course.stats).forEach(([key, value]) => {
    const node = document.querySelector(`#stat-${key}`);
    if (node) node.textContent = String(value).padStart(2, "0");
  });

  const chapterHtml = course.chapters.map((chapter) => `
    <details class="course-chapter" ${chapter.number <= 1 ? "open" : ""}>
      <summary>
        <span class="chapter-number">${String(chapter.number).padStart(2, "0")}</span>
        <span><strong>${escapeHtml(chapter.title)}</strong><small>${chapter.sections.length} 个章节</small></span>
        <span class="chapter-arrow">展开</span>
      </summary>
      <div class="chapter-sections">
        ${chapter.sections.map((section) => `
          <article class="course-section" data-search="${escapeHtml(`${chapter.title} ${section.title} ${section.activities.map((item) => item.title).join(" ")}`.toLowerCase())}">
            <a class="section-main" href="reader.html?article=${encodeURIComponent(section.id)}">
              <span class="type-badge ${section.type}">${typeLabels[section.type]}</span>
              <span class="section-copy">
                <strong>${escapeHtml(section.title)}</strong>
                <small>${section.video_count} 段视频 transcript · ${section.pdf_count} 份 PDF 已融合</small>
              </span>
              <span class="section-open">阅读 →</span>
            </a>
            ${section.activities.length ? `<div class="activity-links">${section.activities.map((activity) => `
              <a href="reader.html?article=${encodeURIComponent(activity.id)}">
                <span class="type-badge ${activity.type}">${typeLabels[activity.type]}</span>
                ${escapeHtml(activity.title)}
              </a>`).join("")}</div>` : ""}
          </article>`).join("")}
      </div>
    </details>`).join("");
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
  const requested = params.get("article") || "week1-intro-ml";
  coursePromise
    .then((course) => curatedSources[requested] || course.articles[requested] || course.articles["week1-intro-ml"])
    .then((article) => {
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
      const toc = document.querySelector("#toc");
      [...articleContent.querySelectorAll("h2, h3")].forEach((heading, index) => {
        heading.id = `section-${index + 1}`;
        const link = document.createElement("a");
        link.href = `#${heading.id}`;
        link.textContent = heading.textContent;
        if (heading.tagName === "H3") link.style.paddingLeft = "24px";
        toc.appendChild(link);
      });
      if (window.MathJax?.typesetPromise) MathJax.typesetPromise([articleContent]);
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
} else {
  coursePromise.then(renderCatalog).catch(() => {
    const catalog = document.querySelector("#course-catalog");
    if (catalog) catalog.innerHTML = '<p class="loading-error">课程目录读取失败。</p>';
  });
}