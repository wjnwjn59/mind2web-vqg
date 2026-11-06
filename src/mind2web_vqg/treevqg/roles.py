"""Small vocabulary layer: HTML tag + attributes -> the words a person would use.

Kept separate from the flat-VQG baseline (``data/vqg.py``) so the TreeVQG package can
evolve — and eventually move — without dragging the old task along.
"""

from __future__ import annotations

from typing import Any

#: Tags that are (usually) actionable by the user.
INTERACTIVE_TAGS: frozenset[str] = frozenset(
    {
        "a",
        "button",
        "input",
        "select",
        "textarea",
        "option",
        "summary",
        "details",
        "label",
    }
)

#: ARIA roles that make an element actionable even when the tag is generic.
INTERACTIVE_ROLES: frozenset[str] = frozenset(
    {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "listbox",
        "checkbox",
        "radio",
        "switch",
        "tab",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "option",
        "slider",
        "spinbutton",
        "treeitem",
    }
)

#: Tags that can group other elements (the Level-2 anchors).
CONTAINER_TAGS: frozenset[str] = frozenset(
    {
        "article",
        "aside",
        "dialog",
        "div",
        "dl",
        "fieldset",
        "footer",
        "form",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

#: Human word for a tag (the vocabulary used in questions/answers).
ROLE_BY_TAG: dict[str, str] = {
    "a": "link",
    "button": "button",
    "textarea": "text area",
    "select": "dropdown",
    "option": "option",
    "img": "image",
    "input": "input field",
    "label": "label",
    "form": "form",
    "nav": "navigation menu",
    "table": "table",
    "th": "table header",
    "td": "table cell",
    "tr": "table row",
    "li": "list item",
    "ul": "list",
    "ol": "list",
    "dl": "list",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "header": "header",
    "footer": "footer",
    "main": "main content area",
    "section": "section",
    "article": "article",
    "aside": "sidebar",
    "div": "container",
    "span": "text",
    "p": "paragraph",
    "dialog": "dialog",
    "fieldset": "field group",
    "summary": "expandable header",
    "details": "expandable section",
}

#: ARIA role -> the same human vocabulary (a `<div role="button">` is a button).
ARIA_ROLE_WORDS: dict[str, str] = {
    "button": "button",
    "link": "link",
    "textbox": "text field",
    "searchbox": "search field",
    "combobox": "dropdown",
    "listbox": "list box",
    "checkbox": "checkbox",
    "radio": "radio button",
    "switch": "toggle switch",
    "tab": "tab",
    "menuitem": "menu item",
    "menuitemcheckbox": "menu item",
    "menuitemradio": "menu item",
    "option": "option",
    "slider": "slider",
    "spinbutton": "number field",
    "treeitem": "tree item",
    "navigation": "navigation menu",
    "form": "form",
    "region": "section",
    "group": "group",
    "list": "list",
    "table": "table",
    "grid": "grid",
    "toolbar": "toolbar",
    "dialog": "dialog",
    "menu": "menu",
}

#: ``<input type=...>`` -> the word a person would use for that field.
ROLE_BY_INPUT_TYPE: dict[str, str] = {
    "text": "text field",
    "search": "search field",
    "email": "email field",
    "password": "password field",
    "number": "number field",
    "tel": "phone field",
    "url": "URL field",
    "date": "date field",
    "datetime-local": "date-time field",
    "month": "month field",
    "week": "week field",
    "time": "time field",
    "range": "slider",
    "checkbox": "checkbox",
    "radio": "radio button",
    "file": "file upload",
    "color": "color picker",
    "submit": "button",
    "button": "button",
    "reset": "button",
}

#: Attributes whose value is a human-readable name for the element.
LABEL_ATTRS: tuple[str, ...] = (
    "aria-label",
    "title",
    "alt",
    "placeholder",
    "data-placeholder",
    "value",
)

#: Attributes kept in the serialized tree (everything else is dropped to stay small).
KEPT_ATTRS: tuple[str, ...] = (
    "id",
    "class",
    "role",
    "type",
    "name",
    "href",
    "placeholder",
    "aria-label",
    "title",
    "alt",
    "value",
    "onclick",
    "tabindex",
    "contenteditable",
    "for",
    "bounding_box_rect",
    "data-bbox",
    "bbox",
)


def input_type(attrs: dict[str, Any]) -> str:
    """Normalized ``type`` of an ``<input>`` (``text`` when absent)."""
    return str(attrs.get("type") or "text").strip().lower()


def is_interactive(tag: str, attrs: dict[str, Any]) -> bool:
    """Can the user act on this node?

    A generic ``<div>`` becomes interactive when it carries an ARIA role, a click
    handler, a tab index or a link — which is how modern JS apps expose their widgets.
    """
    tag = tag.lower()
    if tag in INTERACTIVE_TAGS:
        return True
    if str(attrs.get("role") or "").strip().lower() in INTERACTIVE_ROLES:
        return True
    if attrs.get("onclick") or attrs.get("href"):
        return True
    if str(attrs.get("tabindex") or "").strip() not in {"", "-1"}:
        return True
    return str(attrs.get("contenteditable") or "").strip().lower() == "true"


def is_container(tag: str, attrs: dict[str, Any]) -> bool:
    """Can this node group other elements (a Level-2 candidate)?"""
    tag = tag.lower()
    if tag in CONTAINER_TAGS:
        return True
    return str(attrs.get("role") or "").strip().lower() in {
        "form",
        "group",
        "region",
        "list",
        "table",
        "grid",
        "toolbar",
        "navigation",
        "dialog",
        "menu",
    }


def element_role(tag: str, attrs: dict[str, Any] | None = None) -> str:
    """Human word for an element: ``button``, ``link``, ``text field``, ...

    Falls back to the tag name itself so every node has *some* readable role; this is
    what the Level-1/2 answer uses when the element has no visible label.
    """
    attrs = attrs or {}
    tag = (tag or "").strip().lower()
    aria = str(attrs.get("role") or "").strip().lower()
    if aria in ARIA_ROLE_WORDS:
        return ARIA_ROLE_WORDS[aria]
    if tag == "input":
        return ROLE_BY_INPUT_TYPE.get(input_type(attrs), "input field")
    return ROLE_BY_TAG.get(tag, tag or "element")
