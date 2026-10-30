"""无外部资源的独立 HTML 报告。所有用户文字转义，案例图内嵌，报告可单文件查看。"""

import base64
import html
import json
from pathlib import Path


def percent(value):
    return "—" if value is None else f"{value*100:.2f}%"


def write_html_report(report, path):
    escape = lambda value: html.escape(str(value), quote=True)
    report_directory = Path(path).resolve().parent

    def image_source(relative):
        image = (report_directory / relative).resolve()
        if (
            image.is_relative_to(report_directory)
            and image.is_file()
            and image.suffix == ".png"
        ):
            return "data:image/png;base64," + base64.b64encode(
                image.read_bytes()
            ).decode("ascii")
        return ""

    cells = report.get("cells", [])
    rows = []
    bars = []
    classes = []
    cases = []
    for cell in cells:
        name = cell.get("defense_name", cell.get("model_name", ""))
        protocol = cell["protocol"]
        budget = (
            f'{protocol["budget"]*255:g}/255'
            if protocol["norm"] == "Linf"
            else str(protocol["budget"])
        )
        metrics = cell["metrics"]
        detection = cell.get("detection", {})
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(value)}</td>"
                for value in [
                    name,
                    protocol["name"],
                    protocol["norm"],
                    budget,
                    percent(metrics["clean_accuracy"]),
                    percent(metrics["robust_accuracy"]),
                    percent(detection.get("clean_coverage")),
                    percent(detection.get("adversarial_coverage")),
                    percent(detection.get("undetected_error_rate")),
                    cell.get("seconds", ""),
                ]
            )
            + "</tr>"
        )
        label = escape(name + " / " + protocol["name"] + " / " + budget)
        bars.append(
            f'<div class="bar"><span>{label}</span><meter min="0" max="1" value="{metrics["robust_accuracy"]}"></meter><b>{percent(metrics["robust_accuracy"])}</b></div>'
        )
        class_cells = "".join(
            f'<td style="background:rgba(43,137,100,{.06+.65*(item["robust_accuracy"] or 0):.3f})" title="{escape(item["count"])} samples">{percent(item["robust_accuracy"])}</td>'
            for item in metrics["per_class"]
        )
        classes.append(f"<tr><th>{label}</th>{class_cells}</tr>")
        for example in cell.get("examples", []):
            if len(cases) >= 48:
                continue
            images = "".join(
                f'<figure><img src="{escape(image_source(example[kind]))}" alt="{kind}"><figcaption>{kind}</figcaption></figure>'
                for kind in ["original", "adversarial", "processed"]
                if kind in example
            )
            cases.append(
                f'<article><h3>{label} / sample {example["sample_index"]}</h3><div class="case">{images}</div></article>'
            )
    class_names = report.get("classes", []) or [
        str(index)
        for index in range(len(cells[0]["metrics"]["per_class"]) if cells else 0)
    ]
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Aegis Lab 实验报告</title>
<style>body{{font:14px/1.7 system-ui,sans-serif;color:#263e34;background:#f5f8f6;margin:0}}main{{max-width:1300px;margin:40px auto;padding:28px;background:white;border-radius:14px}}h1,h2{{font-weight:600}}.note{{padding:16px;background:#eff6f1}}section{{margin:32px 0;overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}}td,th{{padding:10px;border:1px solid #dde7e0;text-align:left}}th{{background:#f0f5f1}}.bar{{display:flex;gap:16px;align-items:center;margin:12px 0}}.bar span{{width:55%;font-size:12px}}meter{{flex:1}}.case{{display:flex;gap:16px}}figure{{margin:0}}img{{max-width:220px;width:100%;image-rendering:auto}}pre{{white-space:pre-wrap;word-break:break-all;font-size:11px}}small{{color:#708b7b}}</style>
<main><small>AEGIS LAB · 实验报告</small><h1>模型攻防评测报告</h1><p>{escape(report.get('dataset_name',''))} · {escape(report.get('dataset_split',''))} · {report.get('sample_count',0)} 个样本 · {'随机子集' if report.get('subset') else '完整登记划分'}</p>
<section><h2>结果汇总</h2><table><tr>{''.join('<th>'+value+'</th>' for value in ['模型/防御','攻击/模式','范数','预算','正常准确率','鲁棒准确率','正常覆盖率','攻击覆盖率','未检出错误率','耗时(s)'])}</tr>{''.join(rows)}</table></section>
<section><h2>鲁棒准确率对比</h2>{''.join(bars)}</section><section><h2>逐类别热力表</h2><table><tr><th>实验组合</th>{''.join('<th>'+escape(name)+'</th>' for name in class_names)}</tr>{''.join(classes)}</table></section>
<section><h2>样本案例</h2>{''.join(cases) or '<p>暂无成功攻击案例。</p>'}</section><details><summary>完整协议和数值</summary><pre>{escape(json.dumps(report,ensure_ascii=False,indent=2))}</pre></details></main></html>"""
    Path(path).write_text(body, encoding="utf-8")
