import io
import itertools

import docx


def public(obj):
    """Return list of public attributes of obj.
    
    Parameters
    ----------
    obj : object
        Object to inspect public attributes of.
        
    Returns
    -------
    attrs : list of strings
        List of public attributes of object
    """
    return [attr for attr in dir(obj) if not attr.startswith('_')]


def stream_extract(zipfile, *file_indices):
    """Extract files without saving them to disk.
    
    Parameters
    ----------
    zipfile : ZipFile
        ZipFile object to extract info from.
    *file_indices : int
        Indices of the files to extract from zipfile.
    
    Returns
    -------
    files: generator of files
        Generator yielding extracted files.
    """
    
    files = (zipfile.filelist[idx] for idx in file_indices)
    files = (zipfile.read(file) for file in files)
    files = (io.BytesIO(file) for file in files)
    return files


def get_prompt(readme):
    """Return prompt of essay from a docx file.
    
    Parameters
    ----------
    readme : docx file
        Readme file of an essay.
        
    Returns
    -------
    prompt : str
        String of the essay prompt.
    """
    
    doc = docx.Document(readme)
    pred_drop = lambda s: 'Prompt' not in s
    pred_take = lambda s: 'Rubric' not in s
    prompts = (p.text for p in doc.paragraphs)
    prompts = itertools.dropwhile(pred_drop, prompts)
    next(prompts)
    prompts = itertools.takewhile(pred_take, prompts)
    return '\n'.join(prompts)