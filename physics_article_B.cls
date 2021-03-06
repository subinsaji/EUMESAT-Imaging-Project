\LoadClass[a4paper,11pt]{article}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{physics_article_B}[18/05/15 - Article class for reports written by Nottingham Physics students, without biblatex]

\RequirePackage{fancyhdr}
% this package is used to provide and populate the header and footer
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{graphicx}
%package used to include figures as encapsulated postscript
\RequirePackage{hyperref}
\usepackage{url} 
\usepackage[numbers]{natbib}
\usepackage{float}
\usepackage{graphicx}
\usepackage{import}
\usepackage{booktabs}
\usepackage{wrapfig}
\usepackage{siunitx}
\usepackage{array}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{import}
\usepackage{notoccite}

\usepackage[utf8]{inputenc} % Required for inputting international characters
\usepackage[T1]{fontenc} % Output font encoding for international characters
%\usepackage{fouriernc} % Use the New Century Schoolbook font
\usepackage{graphicx}

%\renewcommand{\familydefault}{cmss}
\renewcommand{\familydefault}{\rmdefault}

\setlength{\hoffset}{-0.4mm}
\setlength{\voffset}{-0.4mm}
\setlength{\oddsidemargin}{5mm}
\setlength{\topmargin}{5mm}
\setlength{\headheight}{5mm}
\setlength{\headsep}{5mm}
\setlength{\textheight}{230mm}
\setlength{\textwidth}{150mm}
\setlength{\marginparsep}{10mm}
\setlength{\footskip}{10mm}

\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\pagestyle{fancy}

%\def\studentid#1{\gdef\@studentid{14320580}}
\def\myname#1{\gdef\@myname{Matt\'ea Goalan & Subin Saji}}
%\def\mytitle#1{\gdef\@mytitle{Galaxy Morphology}}

\lhead{}
\chead{}
\rhead{\@myname}
\lfoot{}
\cfoot{\thepage}
\rfoot{}

%\author{\textbf{d\@myname \\School of Physics and Astronomy\\University of Nottingham}
%\title{\@mytitle}