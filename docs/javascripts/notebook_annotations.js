/**
 * notebook_annotations.js
 *
 * mkdocs-jupyter로 렌더링된 노트북 코드 셀에서
 * `# (N)! 어노테이션 텍스트` 패턴을 감지하여
 * Tippy.js를 사용한 완벽한 네이티브 스타일 툴팁으로 변환합니다.
 */

(function () {
  "use strict";

  const ANNOTATION_RE = /#\s*\((\d+)\)!\s*(.*)/;

  function processCodeCell(preEl) {
    const textNodes = [];
    const walker = document.createTreeWalker(
      preEl,
      NodeFilter.SHOW_TEXT,
      null
    );
    let node;
    while ((node = walker.nextNode())) {
      textNodes.push(node);
    }

    textNodes.forEach((textNode) => {
      const text = textNode.textContent;
      const match = text.match(ANNOTATION_RE);
      if (!match) return;

      const num = match[1];
      const annotationText = match[2].trim();
      const beforeAnnotation = text.slice(0, match.index);

      const parent = textNode.parentNode;
      const beforeNode = document.createTextNode(beforeAnnotation);

      // 래퍼
      const wrapper = document.createElement("span");
      wrapper.className = "nb-annotation";

      // 뱃지 (마커)
      const badge = document.createElement("span");
      badge.className = "nb-annotation__badge";
      badge.textContent = num;
      badge.tabIndex = 0;

      wrapper.appendChild(badge);

      parent.insertBefore(beforeNode, textNode);
      parent.insertBefore(wrapper, textNode);
      parent.removeChild(textNode);

      // 간단한 마크다운 파싱 (백틱 -> <code>, ** -> <strong>)
      let parsedHTML = annotationText
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>');

      // Tippy.js를 사용하여 완벽한 네이티브 팝오버 생성
      if (typeof tippy !== 'undefined') {
        tippy(badge, {
          content: parsedHTML,
          allowHTML: true,            // 마크다운 변환 HTML 허용
          trigger: 'click',           // 클릭 시 열림 (Material 어노테이션 기본 동작)
          interactive: true,          // 툴팁 안쪽 클릭/복사 가능
          placement: 'bottom',        // 보통 아래쪽에 위치
          animation: 'shift-away',    // 부드러운 애니메이션
          theme: 'mkdocs-jupyter',    // CSS 스타일 연동용 커스텀 테마
          maxWidth: 320,
          appendTo: document.body,    // 구조적 제약(overflow: hidden) 방지
        });
      }
    });
  }

  function init() {
    const codeBlocks = document.querySelectorAll(
      ".highlight-ipynb pre:not([data-nb-annotated]), .highlight-ipynb.hl-python pre:not([data-nb-annotated])"
    );
    
    codeBlocks.forEach(pre => {
      pre.setAttribute("data-nb-annotated", "true");
      processCodeCell(pre);
    });
  }

  // DOM 로드 완료 시 및 SPA 라우팅 시 초기화 보장
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
  
  if (typeof document$ !== "undefined") {
    document$.subscribe(function() {
      init();
    });
  }
})();
