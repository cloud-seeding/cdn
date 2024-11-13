import os
from urllib.parse import quote

INITIAL_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>My CDN</title>
    <style>
      body { font-family: monospace; }
      pre { margin-left: 20px; }
      a { text-decoration: none; color: #028391; }
      a:hover { text-decoration: underline; }
    </style>
  </head>
  <body>
    <h1>My CDN</h1>
    <div id="file-list"></div>
  </body>
</html>
"""


def generate_structure(path, prefix='', top_level=True):
    structure = []
    try:
        entries = sorted(os.scandir(path), key=lambda e: (
            not e.is_dir(), e.name.lower()))
        for entry in entries:
            if entry.name.startswith('.'):
                continue

            if top_level and not entry.is_dir():
                continue

            rel_path = os.path.relpath(entry.path, '.')
            if entry.is_dir():
                structure.append(f"{prefix}{entry.name}/")
                structure.extend(generate_structure(
                    entry.path, prefix + '│   ', False))
            else:
                encoded_path = quote(rel_path)
                structure.append(
                    f"{prefix}├── <a href='{encoded_path}'>{entry.name}</a>")

        if structure and not top_level:
            structure[-1] = structure[-1].replace('├──', '└──')
    except PermissionError:
        structure.append(f"{prefix}[Permission Denied]")
    return structure


def generate_html_content(folder_structure):
    return "<pre>\n" + "\n".join(folder_structure) + "\n</pre>"


def main():
    folder_structure = generate_structure('.')
    html_content = generate_html_content(folder_structure)

    updated_content = INITIAL_HTML.replace(
        '<div id="file-list"></div>',
        f'<div id="file-list">{html_content}</div>'
    )

    with open('index.html', 'w') as file:
        file.write(updated_content)


if __name__ == "__main__":
    main()
