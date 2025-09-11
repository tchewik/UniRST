import glob
import os
import xml.etree.ElementTree as ET


def find_single_segment_files(
    root_dir: str,
    recursive: bool = True,
    return_list: bool = True,
) -> list[str]:
    """
    Locate .rs3 XML files that contain only ONE <segment> node.

    Parameters
    ----------
    root_dir      : directory to scan
    recursive     : if True, search sub-directories (default True)
    return_list   : if True, return a list; otherwise print paths and
                    return an empty list

    Returns
    -------
    list[str]     : matching file paths (empty if none or return_list=False)
    """
    pattern = "**/*.rs3" if recursive else "*.rs3"
    paths   = glob.glob(os.path.join(root_dir, pattern), recursive=recursive)

    matches = []
    for path in paths:
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            # skip malformed XML
            continue

        if len(tree.findall(".//segment")) == 1:
            if return_list:
                matches.append(path)
            else:
                print(path)

    return matches
