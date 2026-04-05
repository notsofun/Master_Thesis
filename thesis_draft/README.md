# CL Thesis Template

This template is designed for Computational Linguistics theses at the University of Zurich. It provides a structured LaTeX setup suitable for both Bachelor's and Master's theses. Additionally, it can be used for Informatics students doing their Bachelor's or Master's thesis with a CL supervisor. Make sure you use the corresponding "Declaration of Independent Authorship" of your faculty.

The template contains inline comments that should help you to understand how to adapt it to your needs. For general help on TeX, https://www.overleaf.com/learn is a great resource.

Providing complete and correct bibtex entries can be a pain. We recommend using https://vilda.net/s/ryanize-bib/ to check for issues and guidance on fixing them.

## Usage

### Using the Template with Overleaf

1. Click on the "Menu" button in the top-left corner.
2. Select "Copy Project" to create your own editable copy of the template.
3. Rename your project as needed.
4. The main document is already set to `CLthesis.tex`.
5. You can now edit the files and compile the document.

### Compiling Locally

To compile the thesis locally, you'll need a LaTeX distribution installed on your computer (e.g., TeX Live, MiKTeX).

1. Clone this repository or download the files to your local machine.
2. Open a terminal and navigate to the directory containing the files.
3. Run the following commands:

```
   pdflatex CLthesis.tex
   bibtex CLthesis
   pdflatex CLthesis.tex
   pdflatex CLthesis.tex
```

4. This will generate a PDF file named `CLthesis.pdf`.


## Template Structure

- `CLthesis.tex`: Main LaTeX file
- `chapters/`: Directory containing individual chapter files. These files use the [`subfile package` mechanisms](https://www.overleaf.com/learn/latex/Multi-file_LaTeX_projects) to be able to work on a single chapter. Copy and adapt `chapters/sample_chapter.tex` to add more chapters, e.g. for Dataset or Methods section.
- `images/`: Directory for storing images
- `biblio/`: Directory for bibliography file
- `appendix/`: Directory for appendix files (e.g., the Declaration of Independent Authorship)
