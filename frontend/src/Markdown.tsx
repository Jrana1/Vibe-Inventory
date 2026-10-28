import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function Markdown({ children }: Readonly<{ children: string }>) {
  return <div className="ai-markdown"><ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml
    disallowedElements={["img", "iframe", "script", "style", "input"]}
    components={{ a: ({ children: text, href }) => <a href={href} target="_blank" rel="noopener noreferrer">{text}</a> }}>
    {children}
  </ReactMarkdown></div>;
}
