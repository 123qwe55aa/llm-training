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
  .replace(/`([^`]+)`/g, "<code>$1</code>")
  .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
  .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');

const renderMarkdown = (markdown) => {
  const normalized = markdown
    .replaceAll("No playable video sources found.", "")
    .replaceAll("Your browser does not support this video format. Try using a different browser.", "")
    .replaceAll("0:00 / 0:00", "")
    .replace(/\[mathjaxinline\](.*?)\[\/mathjaxinline\]/gs, "$$$1$$")
    .replace(/\[mathjax\](.*?)\[\/mathjax\]/gs, "$$$$$1$$$$");
  const lines = normalized.replace(/\r/g, "").split("\n");
  const html = [];
  let paragraph = [];
  let listType = "";
  let code = [];
  let inCode = false;
  let inTable = false;

  const closeParagraph = () => {
    if (paragraph.length) {
      const text = paragraph.join(" ");
      if (text.startsWith("**Example:**")) {
        html.push(`<div class="example-block"><p>${inline(text)}</p></div>`);
      } else {
        html.push(`<p>${inline(text)}</p>`);
      }
    }
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
        html.push(`<pre class="algorithm-box"><code>${escapeHtml(code.join("\n"))}</code></pre>`);
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
      const content = line.slice(2);
      if (content.startsWith("**Study Question:**")) {
        html.push(`<blockquote class="study-question">${inline(content.replace("**Study Question:**", "").trim())}</blockquote>`);
      } else {
        html.push(`<blockquote>${inline(content)}</blockquote>`);
      }
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
  if (inCode) html.push(`<pre class="algorithm-box"><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  return html.join("\n");
};

const activityToMarkdown = (text) => {
  const cleaned = text
    .replaceAll("Submit View Answer Ask for Help", "")
    .replaceAll("Run Code Submit View Answer Ask for Help", "")
    .replaceAll("Check Syntax Submit View Answer Ask for Help", "")
    .replaceAll("Submit View Answer", "")
    .replaceAll("You have infinitely many submissions remaining.", "");
  return cleaned.split("\n").map((line) => {
    const value = line.trimEnd();
    if (/^\d+\)\s+\S/.test(value)) return `## ${value}`;
    if (/^\d+\.\d+\)\s+\S/.test(value)) return `### ${value}`;
    if (/^\d+\.\d+[a-z]\)\s+\S/i.test(value)) return `#### ${value}`;
    if (/^(import |from |def |return |class |>>> )/.test(value.trim())) return `\`${value.trim()}\``;
    return value;
  }).join("\n");
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
    kicker: "WEEK 1 · 原版练习册",
    deck: "保留原始题目与代码模板，不包含答案。",
    url: "../downloads/MITx-6.036-Week1-exercises/week1_exercises_questions.txt",
    markdown: false,
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
      articleContent.innerHTML = renderMarkdown(article.markdown ? text : activityToMarkdown(text));
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
