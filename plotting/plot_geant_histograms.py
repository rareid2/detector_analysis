# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:05:34 2015

@author: adotti
"""
import numpy as np
from copy import deepcopy

# DoSSiER OBJECT
FNAL_FMT = {
    "trid": None,
    "testlnk": -1,
    "referencelnk": -1,
    "mcdetaillnk": -1,
    "beamlnk": -1,
    "targetlnk": -1,
    "observablelnk": -1,
    "secondarylnk": -1,
    "reactionlnk": -1,
    "datatable": {
        "dtid": None,
        "datatypeslnk": -1,
        "title": "",
        "npoints": -1,
        "nbins": [],
        "axisTitle": [],
        "val": [],
        "errStatPlus": [],
        "errStatMinus": [],
        "binMin": [],
        "binMax": [],
        "errSysPlus": [],
        "errSysMinus": [],
    },
    "imageblobslnk": 0,
    "scoreslnk": 0,
    "accesslnk": 0,
    "parnames": [],
    "parvalues": [],
    "modtime": None,
}

FNAL_BASE_URL = "http://g4devel.fnal.gov:8080/ValidationWebAPI/webresources/validationWebapi/json/result/"


# Exception used here
class HistoToolException(Exception):
    """
    Base class of all exceptions of this module
    """

    pass


class HistoToolIOException(HistoToolException):
    """
    I/O related exceptions
    """

    pass


class HistoToolDataException(HistoToolException):
    """
    Data format exceptions
    """

    pass


class CannotDownload(HistoToolException):
    """
    Cannot download from DoSSiER Exception
    """

    pass


class MetaDataError(HistoToolDataException):
    """
    Bad meta-data format exception
    """

    pass


class ObjectNotSupported(HistoToolDataException):
    """
    Object type not supported
    """

    pass


class ReshapeError(HistoToolDataException):
    """
    Processing of internal format failed
    """

    pass


class NotAvailableYet(HistoToolException):
    """
    Missing feature.
    """

    pass


class CSVReadError(HistoToolIOException):
    """
    Cannot read CSV file
    """

    pass


class TFileNotOpen(HistoToolIOException):
    """
    Cannot read ROOT file
    """

    pass


class EmptyInput(HistoToolIOException):
    """
    No valid input in input files
    """

    pass


# Utility functions
def _reshape_1d(data):
    """
    Not to be used by client code, for internal use
    Let's convert everything as it would be a 2D object.
    bins= [ [x{0,0},x{0,1},x{0,2},x{0,3}],[x{1,0},x{1,1},x{1,2},x{1,3}] ]
    Where x{i,j} is actually a list of values, for a 1D histo the object will
    look like [ [x{0,0}],[x{1,0}],[x{2,0}] ]
    """
    data["low"] = np.array(data["low"])
    data["high"] = np.array(data["high"])
    data["nb"] = np.array(data["nb"])
    data["bins"] = np.array(data["bins"])
    # "Upgrade" the 1D histo to a 2D structure (with only 1 bin in y)
    matrixf = deepcopy(data["bins"].reshape([1] + list(data["bins"].shape)))
    data["bins"] = matrixf[:, 1:-1, :]
    data["overflow"] = deepcopy(matrixf)
    data["overflow"][:, 1:-1, :] -= data["bins"]
    data["high"] = np.append(data["high"], [0])
    data["low"] = np.append(data["low"], [0])
    data["nb"] = np.append(data["nb"], [1])


def _reshape_2d(data):
    """
    Manipulate data to get a matrix representation
    Not to be used by client code, for internal use.
    """
    data["low"] = np.array(data["low"])
    data["high"] = np.array(data["high"])
    data["nb"] = np.array(data["nb"])
    data["bins"] = np.array(data["bins"])
    matrixf = deepcopy(
        data["bins"].reshape(data["nb"][0] + 2, data["nb"][1] + 2, len(data["bins"][0]))
    )
    data["bins"] = matrixf[1:-1, 1:-1, :]
    data["overflow"] = deepcopy(matrixf)
    data["overflow"][1:-1, 1:-1, :] -= data["bins"]


#########################################################################
## Conversion to/from DoSSiER format
#########################################################################


def convert_to_dossier_histo(histo_bins, dim, bin_edges):
    """
    Convert to DoSSiER format from internal format ('HISTO' type)
    :param histo_bins: Internal fomrat 'bins' content, w/o NaN's
    :param dim: Dimension of the object
    :param bin_edges: tuple with bin edges (min,max)
    :returns: Converted object in DoSSiER format
    """
    # Handle Histograms case, in FNALdb dattypeslnk==1
    # TODO: ASK HANS IF IT IS 1 OR 2
    if len(bin_edges) != 2:
        raise HistoToolDataException(
            "Wrong format of input: pass tuple" " of len 2 for bin_edges"
        )
    _cnv = deepcopy(FNAL_FMT)
    _cnv["datatable"]["datatypeslnk"] = 2
    _cnv["datatable"]["binMin"] = bin_edges[0].tolist()
    _cnv["datatable"]["binMax"] = bin_edges[1].tolist()
    _cnv["datatable"]["val"] = (
        histo_bins[:, :, 1].reshape(histo_bins[:, :, 1].size).tolist()
    )
    _cnv["datatable"]["nbins"].append(histo_bins.shape[1])
    if dim > 1:
        _cnv["datatable"]["nbins"].append(histo_bins.shape[0])
    _entries = np.sum(histo_bins[:, :, 0])
    _errors = (
        np.sqrt(histo_bins[:, :, 2] - histo_bins[:, :, 1] ** 2 / _entries)
        .reshape(histo_bins[:, :, 0].size)
        .tolist()
    )
    _cnv["datatable"]["errStatPlus"] = _cnv["datatable"]["errStatMinus"] = _errors
    return _cnv


def convert_to_dossier_graph(xy_bins, npoints):
    """
    Convert to DoSSiER format from internal format ('XY' type)
    :param xy_bins: Internal fomrat 'bins' content, w/o NaN's
    :param npoints: Number of points
    :returns: Converted object in DoSSiER format
    """
    _cnv = deepcopy(FNAL_FMT)
    _cnv["datatable"]["datatypeslnk"] = 1000
    _cnv["datatable"]["npoints"] = npoints
    # Concatenate X and Y values
    _cnv["datatable"]["val"] = np.append(xy_bins[:, :, 0], xy_bins[:, :, 3]).tolist()
    _cnv["datatable"]["errStatMinus"] = np.append(
        xy_bins[:, :, 1], xy_bins[:, :, 4]
    ).tolist()
    _cnv["datatable"]["errStatPlus"] = np.append(
        xy_bins[:, :, 2], xy_bins[:, :, 5]
    ).tolist()
    return _cnv


def convert_to_dossier(histo, metadata={}):
    """
    Convert to FNALdb format from internal formant
    """
    _bins = np.nan_to_num(histo["bins"])
    if histo["type"] == "HISTO":
        _cnv = convert_to_dossier_histo(_bins, histo["dim"], histo["bin_edges"])
    elif histo["type"] == "XY":
        _cnv = convert_to_dossier_graph(_bins, histo["nbins"])
    else:
        raise ObjectNotSupported(
            "ERROR: Cannot convert object of " "type %s to FNAL format" % histo["type"]
        )
    # Common meta-data to all object types
    if "title" in histo:
        _cnv["datatable"]["title"] = histo["title"]
    if "xaxistit" in histo:
        _cnv["datatable"]["axisTitle"].append(histo["xaxistit"])
    if "yaxistit" in histo:
        _cnv["datatable"]["axisTitle"].append(histo["yaxistit"])
    for _k, _v in metadata.items():
        if _k not in _cnv:
            print("WARNING: Adding metadata %s not in the FNALdb list of fields." % _k)
            print("         This may cause a failure in uploading the results.")
        _cnv[_k] = _v
    return _cnv


def convert_from_dossier_graph(histo, errors):
    """
    Convert from DoSSiER format for XY type object
    :param histo: the 'datatable' property of dossier objects
    :param erros: errors tuple for x and y
    :returns: internal format object
    """
    _rsl = {"type": "XY", "dim": 1}
    if histo["datatypeslnk"] == 1000:
        # Both stat and sys errors
        print("WARNING: Sys and Stat errors added as sqrt(a**2+b**2)")
    _rsl["quantities"] = [
        "x",
        "err_x_low",
        "err_xhistoigh",
        "y",
        "err_y_low",
        "err_yhistoigh",
    ]
    # Add 2 points for the fake overflow and underflow
    _rsl["nbins"] = histo["npoints"] + 2
    _rsl["nb"] = [histo["npoints"]]
    if len(h["axisTitle"]) == 1:
        histo["axisTitle"] += [""]
    _rsl["xaxistit"], _rsl["yaxistit"] = histo["axisTitle"]
    _rsl["bins"] = [[0, 0, 0, 0, 0, 0]]
    nump = histo["npoints"]
    x, err_x_low, err_xhistoigh = (
        histo["val"][:nump],
        errors[0][:nump],
        errors[1][:nump],
    )
    y, err_y_low, err_yhistoigh = (
        histo["val"][nump:],
        errors[0][nump:],
        errors[1][nump:],
    )
    data = np.vstack((x, err_x_low, err_xhistoigh, y, err_y_low, err_yhistoigh)).T
    _rsl["bins"] += data.tolist()
    _rsl["low"], _rsl["high"] = [np.min(x)], [np.max(x)]
    _rsl["bins"].append([0, 0, 0, 0, 0, 0])
    return _rsl


def convert_from_dossier_histo(histo, errors):
    """
    Convert from DoSSiER format for XY type object
    :param histo: the 'datatable' property of dossier objects
    :param erros: errors tuple for x and y
    :returns: internal format object
    """
    _rsl = {"type": "HISTO", "dim": len(histo["nbins"])}
    if h["datatypeslnk"] == 1:
        # Both stat and sys errors
        print("WARNING: Sys and Stat errors added as sqrt(a**2+b**2)")
    if _rsl["dim"] > 2:
        raise ObjectNotSupported(
            "ERROR: Objects of dimension larger than 2 not supported"
        )
    if not np.array_equal(errors[0], errors[1]):
        raise ObjectNotSupported(
            "ERROR: Histogram with LOW error != HIGH error not supported"
        )
    _tot_err = errors[1]
    # Reshape bins edges, values and errors to the correct dimensions
    reshape = lambda a: np.array(a).reshape(histo["nbins"], order="F").T
    _vals, _err, _binMin, _binMax = list(
        map(reshape, (histo["val"], _tot_err, histo["binMin"], histo["binMax"]))
    )
    _rsl["bin_edges"] = (_binMin, _binMax)
    _rsl["quantities"] = ["entries", "Sw", "Sw2", "Sxw0", "Sx2w0"]
    if _rsl["dim"] == 2:
        _rsl["quantities"] += ["Sxw1", "Sx2w1"]
    # Add fake underflow
    _rsl["bins"] = [[0] * len(_rsl["quantities"])]
    # This is the total number of points, including overflows
    # At each dimension, add 2 for O/U flows
    _rsl["nbins"] = np.prod(np.array(histo["nbins"]) + 2)
    _entriesPerBin = np.array(
        [np.sum(_vals) / np.prod(histo["nbins"])] * len(histo["val"])
    )
    _entriesPerBin = reshape(_entriesPerBin)
    _zeros = np.zeros_like(_vals)
    if _rsl["dim"] == 1:
        data = np.vstack((_entriesPerBin, _vals, _err**2, _zeros, _zeros)).T
    elif _rsl["dim"] == 2:
        data = np.vstack(
            (_entriesPerBin, _vals, _err**2, _zeros, _zeros, _zeros, _zeros)
        ).T
    else:  # Never get here
        assert False
    _rsl["nb"] = np.array(histo["nbins"])
    _rsl["low"] = np.min(
        histo["binMin"]
    )  # ... Need to take into consideration this is array for dim>1
    _rsl["high"] = np.max(histo["binMax"])  # ... Same as before
    _rsl["bins"] += data.tolist()
    _rsl["bins"].append([0] * len(_rsl["quantities"]))
    return _rsl


def convert_from_dossier(histo):
    """
    Convert to internal format from DoSSiER format
    :param histo: dictionary in DoSSiER format
    :return: internal format dictionary
    """
    # Get rid of metadata
    _h = histo["datatable"]
    _rsl = {
        "title": h["title"],
        "annotations": [],
    }
    # Treat case sys errors are missing
    # Clean up data, if Sys errors are missing, create a 0 array
    if "errSysPlus" not in _h:
        _h["errSysPlus"] = []
    if "errSysMinus" not in _h:
        _h["errSysMinus"] = []
    if len(_h["errSysMinus"]) == 0:
        _h["errSysMinus"] = np.zeros(len(_h["val"]))
    if len(_h["errSysPlus"]) == 0:
        _h["errSysPlus"] = np.zeros(len(_h["val"]))
    # TODO ask Hans about this case, I interpret it as errStatPlus==errStatMinus
    if len(_h["errStatPlus"]) == 0:
        _h["errStatPlus"] = _h["errStatMinus"]
    _ck = len(_h["val"])
    assert (
        _ck == len(_h["errStatMinus"])
        and _ck == len(_h["errStatPlus"])
        and _ck == len(_h["errSysMinus"])
        and _ck == len(_h["errSysPlus"])
    ), "Format error, size of errors and values arrays do not match"
    tot_err_low = np.sqrt(
        np.array(_h["errStatMinus"]) ** 2 + np.array(_h["errSysMinus"]) ** 2
    )
    tot_err_high = np.sqrt(
        np.array(_h["errStatPlus"]) ** 2 + np.array(_h["errSysPlus"]) ** 2
    )

    if _h["datatypeslnk"] >= 1000 and _h["datatypeslnk"] <= 1001:
        _rsl.update(convert_from_dossier_graph(_h, (tot_err_low, tot_err_high)))
    elif _h["datatypeslnk"] == 1 or _h["datatypeslnk"] == 2:
        _rsl.update(convert_from_dossier_histo(_h, (tot_err_low, tot_err_high)))
    else:
        raise ObjectNotSupported(
            "ERROR: Unknown object of FNAL db " "type: %s" % _h["datatypeslnk"]
        )
    if _rsl["dim"] == 1:
        _reshape_1d(_rsl)
    elif _rsl["dim"] == 2:
        _reshape_2d(_rsl)
    return _rsl


###############################################################################
## Conversion to/from CSV format
###############################################################################


# TODO: Handle non equidistant bins
def convert_to_csv(histo):
    """
    Convert to CSV G4analysis format from internal format
    """
    if histo["type"] != "HISTO":
        # TODO: implement
        raise ObjectNotSupported(
            "ERROR: Conversion to CSV not supported for object of type: %s"
            % histo["type"]
        )
    header = """#class tools::histo::h{dim}d
#title {title}
#dimension {dim}
"""
    output = header.format(**histo)
    output += "#axis edges "
    _lowerEdges = histo["bin_edges"][0]
    _upperEdges = histo["bin_edges"][1]
    _nbX = histo["nb"][0]
    for _e in _lowerEdges[:_nbX]:
        output += " {0}".format(_e)
    # Add extra bin edge (upper limit)
    output += " {0}\n".format(_upperEdges[_nbX])
    if histo["dim"] > 1:
        _nbY = histo["nb"][1]
        output += "#axis edges "
        for _e in _lowerEdges[_nbX + 1 : _nbX + 1 + _nbY]:
            output += " {0}".format(_e)
        output += " {0}\n".format(_upperEdges[_nbX + 1 + _nbY])
        if histo["dim"] > 2:
            output += "#axis edges "
            for _e in _lowerEdges[_nbX + _nbY + 2 :]:
                output += " {0}".format(_e)
            output + " {0}\n".format(_upperEdges[_nbX + _nbY + 2])
            if histo["dim"] > 3:
                raise ObjectNotSupported(
                    "ERROR: Conversion to CVS not supported for object of dimention: %s"
                    % histo["dim"]
                )
    if "yaxistit" not in histo:
        histo["yaxistit"] = ""
    if histo["dim"] > 2 and "zaxistit" not in histo:
        histo["zaxistit"] = ""
    output += (
        "#annotation axis_x.title {xaxistit}\n"
        "#annotation axis_y.title {yaxistit}\n"
        "#annotation axis_x.title {zaxistit}\n"
        "#bin_number {nbins}\n".format(**histo)
    )
    for ann in histo["annotations"]:
        output += "#annotation %s" % ann
    output += str(histo["quantities"]).lstrip("[").rstrip("]").replace("'", "")
    output += "\n"
    from copy import deepcopy

    matrixf = deepcopy(histo["overflow"])
    if histo["dim"] == 1:
        matrixf[:, 1:-1, :] = histo["bins"]
        bins = matrixf.reshape(matrixf.shape[1], matrixf.shape[2])
    if histo["dim"] == 2:
        matrixf[1:-1, 1:-1, :] = histo["bins"]
        bins = matrixf.reshape((matrixf.shape[0] * matrixf.shape[1], matrixf.shape[2]))
    for row in bins:
        output += str(row.tolist()).lstrip("[").rstrip("]")
        output += "\n"
    return output


def convert_from_csv(lines, name=""):
    """
    Convert to internal format from G4Analysis CSV format.
    @lines is a list of ascii lines
    @return dictionary representing histogram
    """
    data = {
        "nb": [],
        "low": [],
        "high": [],
        "annotations": [],
        "type": "HISTO",
        "name": name,
        "bin_edges": (np.array([]), np.array([])),
    }
    import re

    # TODO: add histo::h3d type
    magicline = re.compile(r"#class tools::histo::[p,h][1,2]d")
    histotit = re.compile(r"#title (?P<title>.*)\n$")
    dimension = re.compile(r"#dimension (?P<dim>\d+)\n$")
    xaxis_tit = re.compile(r"#annotation axis_x\.title(?P<xaxistit>.*)\n$")
    yaxis_tit = re.compile(r"#annotation axis_y\.title(?P<yaxistit>.*)\n$")
    zaxis_tit = re.compile(r"#annotation axis_z\.title(?P<zaxistit>.*)\n$")
    annotation = re.compile(r"#annotation (?P<prop>[^\s]+)(?P<content>.*)\n$")
    nbins = re.compile(r"#bin_number (?P<nbins>\d+)\n$")
    # For fixed binning: fixed nbins min max
    axis_f = r"(?P<nb>\d+) (?P<low>-?\d+\.?\d*[e]?[-\+]?\d*) (?P<high>-?\d+\.?\d*[e]?[-\+]?\d*)"
    # For variable binning: edges [list of edges]
    axis_e = r"(?P<binning>(-?\d+\.?\d*[e]?[-\+]?\d*\s?)+)"
    axis = re.compile(r"#axis (?P<type>(fixed|edges)) (%s|%s)\n$" % (axis_f, axis_e))
    noheader = re.compile(r"[^#].*")  # Not header
    # re.compile(r'^.*(?P<dim>\d+D) histogram \d+: (?P<title>.*)\n$')
    # If first line is not the magic one, skip
    correct = magicline.match(lines[0])
    if not correct:
        raise CSVReadError("File does not seem a valid g4analysis histo")

    def matchmetadata(line, expr, content):
        match = expr.match(line)
        if match:
            return match.group(content)
        else:
            return None

    startline = 1
    _fixedBinning, _edgesBinning = False, False
    for lin in lines[1:]:
        found = False
        # Deal with axis:
        match = axis.match(lin)
        if match:
            if match.group("type") == "fixed":
                _fixedBinning = True
                _nb, _low, _high = (
                    int(match.group("nb")),
                    float(match.group("low")),
                    float(match.group("high")),
                )
                _binWidth = (_high - _low) / _nb
                _bins = np.append(
                    np.linspace(start=_low, stop=_high, num=_nb),
                    np.array(_high + _binWidth),
                )
                data["nb"].append(_nb)
                data["low"].append(_low)
                data["high"].append(_high)
                data["bin_edges"] = (
                    np.append(data["bin_edges"][0], _bins[:-1]),
                    np.append(data["bin_edges"][1], _bins[1:]),
                )
            elif match.group("type") == "edges":
                _edgesBinning = True
                _bins = np.array([float(x) for x in match.group("binning").split()])
                data["nb"].append(
                    _bins.size - 1
                )  # Bin edges include lower and upper limit
                data["low"].append(_bins.min())
                data["high"].append(_bins.max())
                data["bin_edges"] = (
                    np.append(data["bin_edges"][0], _bins[:-1]),
                    np.append(data["bin_edges"][1], _bins[1:]),
                )
        for k, exp in [
            ("title", histotit),
            ("dim", dimension),
            ("nbins", nbins),
            ("xaxistit", xaxis_tit),
            ("yaxistit", yaxis_tit),
            ("zaxistit", zaxis_tit),
        ]:
            vvv = matchmetadata(lin, exp, k)
            if vvv:
                found = True
                try:
                    _dat = float(vvv)
                except:
                    _dat = vvv.strip()  # Not a number
                if k in ("nb", "low", "high"):
                    data[k].append(_dat)
                else:
                    data[k] = _dat
        if not found:
            # All the rest is probably an annotation
            # Annotations can be multiple
            annot_k = matchmetadata(lin, annotation, "prop")
            annot_v = matchmetadata(lin, annotation, "content")
            if annot_k and annot_v:
                data["annotations"].append((annot_k, annot_v))
        # First line past header
        if noheader.match(lin):
            break
        startline += 1
    data["dim"] = int(data["dim"])
    if _fixedBinning & _edgesBinning:
        raise ObjectNotSupported(
            "ERROR: No support for mixed 'fixed' and 'edges' binning scheme"
        )
    # Done with header parsing, parse content of histogram
    quantities = re.findall(
        r"([^,]+)(?:,|$)", lines[startline]
    )  # Split line of quantities
    quantities[-1] = quantities[-1].replace("\n", "")
    data["quantities"] = quantities
    bins = []
    for lin in lines[startline + 1 :]:
        bins.append([float(a) for a in re.findall(r"([^,]+)(?:,|$)", lin)])
    # The format is the following:
    # There are to consider
    # underflows and overflows. In CSV format for each bin along an axis there
    # are underflow and overflow bins. So for example, consider a 2D histo with
    # 2 bins ber axis. The 2D will look like:
    #    7.1 |  5.1 | 5.2  | 7.4
    #   -----===============----
    #    3.1 || 1.1 | 2.1 || 4.1
    #   ------------------------
    #    3.2 || 1.2 | 2.2 || 4.2
    #   -----===============---
    #    7.2 |  6.1 | 6.2  | 7.3
    # Where Bin{1,1}=1.2 Bin{1,2}=1.2 Bin{2,1}=2.1 Bin{2,2}=2.2
    #       Bin{U,U}=7.1 Bin{O,U}=7.4 Bin{U,O}=7.2 Bin{O,O}=7.3
    #       Bin{U,1}=3.1 Bin{U,2}=3.2 Bin{O,1}=4.1 Bin{O,2}=4.2
    #       Bin{1,U}=5.1 Bin{2,U}=5.2 Bin{1,O}=6.1 Bin{2,O}=6.2
    # Remember that each bin has several quantites
    data["bins"] = bins
    if data["dim"] == 1:
        _reshape_1d(data)
    elif data["dim"] == 2:
        _reshape_2d(data)
    else:
        raise ReshapeError("ERROR: cannot read object")
    return data


#########################################################################
## Conversion from/to ROOT
#########################################################################


def convert_to_root(histo):
    """
    Convert to ROOT format from internal format
    :param histo: internal format object
    :returns: converted object in ROOT format
    """
    if histo["type"] == "XY":
        return convert_to_root_tgraph(histo)
    elif histo["type"] == "HISTO":
        return convert_to_root_histo(histo)
    else:
        print()
        raise ObjectNotSupported("ERROR: object type not supported %s" % histo["type"])


def convert_to_root_histo(histo):
    """
    Convert to ROOT format from internal HISTO
    :param histo: internal format object
    :returns: covnerted object in ROOT format
    """
    if histo["dim"] == 1:
        return convert_to_root_h1(histo)
    elif histo["dim"] == 2:
        return convert_to_root_h2(histo)
    else:
        raise ObjectNotSupported("ROOT histograms of 1 or 2 dimensions only")


def convert_to_root_tgraph(histo):
    """
    Convert to ROOT.TGraph from internal format
    :param histo: internal format object
    :returns: a ROOT.TGraph object
    """
    assert histo["type"] == "XY" and histo["dim"] == 1
    from ROOT import TGraphAsymmErrors

    _tgr = TGraphAsymmErrors(histo["nbins"])
    _tgr.SetName(histo["name"])
    _tgr.SetTitle(histo["title"])
    _tgr.GetXaxis().SetTitle(histo["xaxistit"])
    _tgr.GetYaxis().SetTitle(histo["yaxistit"])
    _data = histo["bins"]
    for idx in range(0, histo["nbins"]):
        _pt = _data[0][idx]
        _tgr.SetPoint(idx, _pt[0], _pt[3])
        _tgr.SetPointEXlow(idx, _pt[1])
        _tgr.SetPointEXhigh(idx, _pt[2])
        _tgr.SetPointEYlow(idx, _pt[4])
        _tgr.SetPointEYhigh(idx, _pt[5])
    return _tgr


def convert_to_root_h1(histo):
    """
    Convert to ROOT.TH1D from internal format
    :param histo: internal format object
    :returns: a ROOT.TH1D object
    """
    assert histo["type"] == "HISTO" and histo["dim"] == 1
    _edg = np.append(histo["bin_edges"][0], histo["bin_edges"][1][-1])
    import ctypes

    _root_edg = len(_edg) * ctypes.c_double
    _root_edg = _root_edg(*_edg)
    from ROOT import TH1D

    h = TH1D(histo["name"] + "_converted", histo["title"], histo["nb"][0], _root_edg)
    h.GetXaxis().SetTitle(histo["xaxistit"])
    h.GetYaxis().SetTitle(histo["yaxistit"])
    _data = histo["bins"]
    _entries = np.sum(_data[0, :, 0])
    for idx in range(1, histo["nb"][0] + 1):  # ROOT bins start from 1
        bin_data = _data[0][idx - 1]
        h.SetBinContent(idx, bin_data[1])
        from math import sqrt

        _error = sqrt(bin_data[2] - bin_data[1] ** 2 / _entries)
        h.SetBinError(idx, _error)
    # Note: Under/Over-flow are lost, DoSSiER does not store them!
    return h


def convert_to_root_h2(histo):
    """
    Convert to ROOT.TH2D from internal format
    :param histo: internal format object
    :returns: a ROOT.TH2D object
    """
    assert histo["type"] == "HISTO" and histo["dim"] == 2
    raise NotAvailableYet("ERROR: conversion to ROOT::TH2D not yet implemented")


def convert_from_root(h):
    """
    Try to convert a ROOT object to internal format
    @h is the input ROOT object
    @return the converted object
    @raise ObjectNotSupported if object is not one of the supported objects
    """
    if h.InheritsFrom("TH3"):
        # TODO: implement
        raise NotAvailableYet("TH3 not supported yet")
    if h.InheritsFrom("TH1"):
        return convert_from_root_th1(h)
    elif h.InheritsFrom("TGraph"):
        return convert_from_root_tgraph(h)
    raise ObjectNotSupported("Object type unknown")


def convert_from_root_tgraph(g):
    """
    Convert to internal format from ROOT format
    @h is a ROOT.TGraph object
    @return dictionary representing graph
    """
    data = {}
    data["name"] = g.GetName()
    data["type"] = "XY"
    data["title"] = g.GetTitle()
    data["dim"] = 1
    data["nbins"] = g.GetN()
    data["nb"] = [g.GetN()]
    data["xaxistit"] = g.GetXaxis().GetTitle()
    data["yaxistit"] = g.GetYaxis().GetTitle()
    # Specify bins X-high X-low one by one
    import ctypes

    x, y = ctypes.c_double(), ctypes.c_double()
    _lows = []
    _highs = []
    data["bins"] = [[0, 0, 0, 0, 0, 0]]  # Underflows
    for n in range(data["nb"][0]):
        g.GetPoint(n, x, y)
        # In case there is no error we use 0 as error,
        # the ROOT method will return -1
        exl = np.max([g.GetErrorXlow(n), 0])
        exh = np.max([g.GetErrorXhigh(n), 0])
        eyl = np.max([g.GetErrorYlow(n), 0])
        eyh = np.max([g.GetErrorYhigh(n), 0])
        _lows.append(x.value - exl)
        _highs.append(x.value + exh)
        data["bins"].append([x.value, exl, exh, y.value, eyl, eyh])
    data["low"] = [np.min(_lows)]
    data["high"] = [np.max(_highs)]
    data["bins"].append([0, 0, 0, 0, 0, 0])  # Overflow
    data["quantities"] = [
        "x",
        "err_x_low",
        "err_x_high",
        "y",
        "err_y_low",
        "err_y_high",
    ]
    _reshape_1d(data)
    return data


def convert_from_root_th1(h):
    """
    Convert to internal fomrat from ROOT format
    @h is a ROOT.TH1 object
    @return dictionary representing histogram
    """
    data = {}
    data["name"] = h.GetName()
    data["type"] = "HISTO"
    data["title"] = h.GetTitle()
    data["dim"] = 1
    xdim = h.GetNbinsX() + 2  # include Under(Over)flow
    ydim = zdim = 1
    if h.InheritsFrom("TH2"):
        data["dim"] = 2
        ydim = h.GetNbinsY() + 2
        if h.InheritsFrom("TH3"):
            zdim = h.GetNbinsZ() + 2
            data["dim"] = 3
    data["nbins"] = xdim * ydim * zdim
    data["nb"] = [h.GetNbinsX()]
    data["low"] = [h.GetXaxis().GetBinLowEdge(1)]  # 0 is underflow (I think)
    data["high"] = [h.GetXaxis().GetBinUpEdge(h.GetNbinsX())]
    if data["dim"] == 2:
        data["nb"] += [h.GetNbinsY()]
        data["low"] += [h.GetYaxis().GetBinLowEdge(1)]  # 0 is underflow (I think)
        data["high"] += [h.GetYaxis().GetBinUpEdge(h.GetNbinsY())]
    if data["dim"] == 3:
        data["nb"] += [h.GetNbinsZ()]
        data["low"] += [h.GetZaxis().GetBinLowEdge(1)]  # 0 is underflow (I think)
        data["high"] += [h.GetZaxis().GetBinUpEdge(h.GetNbinsZ())]
    data["xaxistit"] = h.GetXaxis().GetTitle()
    data["yaxistit"] = h.GetYaxis().GetTitle()
    # Ok, this should have more fileds for 2D and 3D, see note later
    data["quantities"] = ["entries", "Sw", "Sw2", "Sxw0", "Sxw02"]
    data["bins"] = []
    binsx = list(range(0, h.GetNbinsX() + 2))  # Include underflow and overflow
    if data["dim"] > 1:
        binsy = list(range(0, h.GetNbinsY() + 2))
        if data["dim"] > 2:
            binsz = list(range(0, h.GetNbinsZ() + 2))
        else:
            binsz = [1]
    else:
        binsy = [1]
        binsz = [1]
    # ROOT does not give me number of entries per bin (e.g. number
    # of times Fill has been called for a given bin).
    # However we need this to calculate error, and we need the sum
    # of entries,
    nbins = float(h.GetNbinsX() * h.GetNbinsY() * h.GetNbinsZ())
    entries = float(h.GetEntries())
    entriesPerBin = float(entries) / nbins
    # Build bin edges arrays
    bEm = []
    bEM = []
    for bin in range(1, h.GetNbinsX() + 1):
        bEm.append(h.GetXaxis().GetBinLowEdge(bin))
        bEM.append(h.GetXaxis().GetBinUpEdge(bin))
    if data["dim"] > 1:
        for bin in range(1, h.GetNbinsY() + 1):
            bEm.append(h.GetYaxis().GetBinLowEdge(bin))
            bEM.append(h.GetYaxis().GetBinUpEdge(bin))
        if data["dim"] > 2:
            for bin in range(1, h.GetNbinsZ() + 1):
                bEm.append(h.GetZaxis().GetBinLowEdge(bin))
                bEM.append(h.GetZaxis().GetBinUpEdge(bin))
    data["bin_edges"] = (np.array(bEm), np.array(bEM))
    bins = []
    for iz in binsz:
        for iy in binsy:
            for ix in binsx:
                bins.append((ix, iy, iz))
    for bin in bins:
        # Serious limitation in ROOT: not possible to distinguish between
        # Number of entries and bin content (i.e. sumW). Only if w=1 when filled
        # The two are the same. For plotting (and upload to FNALDB) we need only
        # SumW.

        # Simple, get the error from ROOT, forget how they are calculated and
        # Calculate the internal format numbers so that when we use it later
        # You get back exactly the same number:
        # We calculate error as: E=Sqrt(Sum(w**2)-Sum(w)**2/N)
        # So let's calculate the Sum(w**2) so that this forumla gives
        # back what root calls error.
        # Sum(w**2)=E**2-Sum(w)**2/N
        _e = h.GetBinError(bin[0], bin[1], bin[2])
        _sw = h.GetBinContent(bin[0], bin[1], bin[2])
        _sumw2 = _sw**2 / float(entries) + _e**2
        if h.InheritsFrom("TProfile") or h.InheritsFrom("TProfile2D"):
            entriesPerBin = h.GetBinEntries(h.GetBin(bin[0], bin[1], bin[2]))
        data["bins"].append([entriesPerBin, h.GetBinContent(*bin), _sumw2, 0, 0])
    if data["dim"] == 1:
        _reshape_1d(data)
    elif data["dim"] == 2:
        _reshape_2d(data)
    else:
        raise ObjectNotSupported("ERROR: dimension > 3 not supported")
    return data


###############################################################################
## Plotting functions
###############################################################################


def plot(histos, fname):
    """
    Plot the histograms
    """
    import matplotlib.pyplot as plt

    figure = plt.figure()
    figure.tight_layout(pad=0.5)
    if type(histos) != type([]):
        histos = [histos]
    cols = 2  # min([len(histos), 3])
    rows = 3  # len(histos) / cols
    # if rows * cols < len(histos):
    #    rows += 1
    ctr = 1
    for histo in histos:
        ax = figure.add_subplot(int(rows), int(cols), ctr)
        ctr += 1
        if histo["type"] == "HISTO":
            if histo["dim"] == 1:
                plot_single_1d(histo, ax, fname)
            elif histo["dim"] == 2:
                plot_single_2d(histo, ax, fname)
            else:
                print(
                    "ERROR: No support for histograms with dim>2, skipping:",
                    histo["title"],
                )
        elif histo["type"] == "XY":
            plot_single_xy(histo, ax)
        else:
            print(
                "ERROR: Cannot plot object of non HISTO type, skipping:", histo["title"]
            )


def plot_single_xy(histo, ax=None):
    import matplotlib.pyplot as plt

    bins = histo["bins"][0]
    _x, _y = bins[:, 0], bins[:, 3]
    _xerr = [bins[:, 1], bins[:, 2]]
    _yerr = [bins[:, 4], bins[:, 5]]
    if ax != None:
        ax.errorbar(_x, _y, xerr=_xerr, yerr=_yerr, fmt="o-", markersize=3)
    else:
        plt.errorbar(_x, _y, xerr=_xerr, yerr=_yerr, fmt="o-", markersize=3)
    if "xaxistit" in histo:
        plt.xlabel(histo["xaxistit"])
    if "yaxistit" in histo:
        plt.ylabel(histo["yaxistit"])
    if "title" in histo:
        plt.title(histo["title"])


def plot_single_1d(histo, ax=None, fname=None):
    import matplotlib.pyplot as plt

    # xb,xl,xh=float(histo['nb'][0]),float(histo['low'][0]),float(histo['high'][0])
    # x=np.arange(xl,xh,(xh-xl)/xb)+(xh-xl)/(2*xb) #Note the parenthesis
    # xerr=np.array([.5*(xh-xl)/xb]*len(x))
    bins = np.array(histo["bins"][0])
    x = (histo["bin_edges"][0] + histo["bin_edges"][1]) / 2
    xerr = (histo["bin_edges"][0] - histo["bin_edges"][1]) / 2

    # Bins: Entries,Sum(W),Sum(W**2),...
    y = bins[:, 1]
    # Error on bin content: Sqrt(Sum W**2 - (Sum W)**2/N)
    _entries = np.sum(bins[:, 0])
    yerr = np.sqrt(bins[:, 2] - bins[:, 1] ** 2 / _entries)
    plt.xlim(x[0] - xerr[0], x[-1] + xerr[-1])
    if ax != None:
        ax.plot(x, y, markersize=3)
    else:
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o-", markersize=3)
    if "xaxistit" in histo:
        plt.xlabel(histo["xaxistit"])
    if "yaxistit" in histo:
        plt.ylabel(histo["yaxistit"])
    if "title" in histo:
        plt.title(histo["title"])
    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    print("finished plotting")
    plt.savefig(fname)

def plot_single_2d(histo, ax=None, fname="test"):
    import matplotlib.pyplot as plt

    # xb,xl,xh=float(histo['nb'][0]),float(histo['low'][0]),float(histo['high'][0])
    # yb,yl,yh=float(histo['nb'][1]),float(histo['low'][1]),float(histo['high'][1])
    # Calculate in one go the [x,y] bin centers and then remember that the edges are
    # serialized
    xy = (histo["bin_edges"][0] + histo["bin_edges"][1]) / 2
    x = xy[: histo["nb"][0]]
    y = xy[histo["nb"][0] :]
    # x=np.arange(xl,xh,(xh-xl)/xb)+(xh-xl)/(2*xb)
    # y=np.arange(yl,yh,(yh-yl)/yb)+(yh-yl)/(2*yb)
    X, Y = np.meshgrid(x, y)
    bins = histo["bins"]
    z = bins[:, :, 1]
    # https://stackoverflow.com/questions/56221246/best-way-to-plot-a-2d-contour-plot-with-a-numpy-meshgrid
    z_grid = z.reshape(len(y), len(x))
    if ax != None:
        cs = ax.contourf(X, Y, z_grid)
        plt.colorbar(cs)
    else:
        cs = plt.contourf(X, Y, z)
        plt.colorbar()
    if "xaxistit" in histo:
        plt.xlabel(histo["xaxistit"])
    if "yaxistit" in histo:
        plt.ylabel(histo["yaxistit"])
    if "title" in histo:
        plt.title(histo["title"])
    plt.ylim([8,14])

    plt.subplots_adjust(wspace=0.4, hspace=0.8)
    print("finished plotting")
    plt.savefig(fname)


###############################################################################
## Interact with FNAL DB
###############################################################################


def download(test_id, timeout=10):
    """
    Download from FNAL db the given result in json format
    @return the json representation
    """
    from urllib.request import urlopen
    from urllib.error import URLError
    from json import loads
    import socket

    try:
        dataf = urlopen(FNAL_BASE_URL + str(test_id), None, timeout)
        return loads(dataf.readlines()[0])
    except URLError as e:
        if isinstance(e.reason, socket.timeout):
            msg = "ERROR: Timeout (%ds) accessing test with ID %d at %s" % (
                timeout,
                test_id,
                FNAL_BASE_URL,
            )
        else:
            msg = "ERROR: Cannot downlaod result with ID %d from %s, reason: %s" % (
                test_id,
                FNAL_BASE_URL,
                e.reason,
            )
    except socket.timeout as e:
        msg = "ERROR: Timeout (%ds) accessing test with ID %d at %s" % (
            timeout,
            test_id,
            FNAL_BASE_URL,
        )
    except:
        msg = "ERROR: Cannot downlaod result with ID %d from %s" % (
            test_id,
            FNAL_BASE_URL,
        )
    raise CannotDownload(msg)


def upload(results):
    """
    Upload result to the FNAL db via web interface
    @results is a JSON list in FNAL db format
    """
    raise NotAvailableYet("LIMITATION: upload not yet implemented")


def generate_metadata_file(filename="metadata.json"):
    """
    Generate skeleton of metadata and save
    it to json file.
    @return dictionary of metadata
    """
    _md = {".*": {}}
    from copy import deepcopy

    _md[".*"] = deepcopy(FNAL_FMT)
    del _md[".*"]["datatable"]  # Remove data and keep mds
    _of = open(filename, "w")
    import json

    json.dump(_md, _of, indent=1)
    _of.close()
    return _md


###############################################################################
## Utility functions used by main
###############################################################################


def _getObjsInTDir(path, obj, hh=[]):
    """
    Recursive read of TObjects in a TFile
    It returns list of full paths of all
    objects in the tfile
    """
    if not obj.InheritsFrom("TDirectoryFile"):
        # Not a TFile or a TDirectory, it is an object
        hh.append(path + obj.GetName())
    else:
        # This is a directory
        path += obj.GetName() + "/"
        _listkeys = obj.GetListOfKeys()
        _key = _listkeys.MakeIterator()
        while _key():
            hh = _getObjsInTDir(path, _key.ReadObj(), hh)
    return hh


def _readMetaDataFile(filename):
    """
    Read in metadata file in both JSON or pickle formats
    """
    from os import path

    _fn, _extension = path.splitext(filename)
    try:
        import pickle

        _file = open(filename, "r")
        _data = pickle.load(_file)
    except:
        # assume is json
        import json

        _file = open(filename, "r")
        _data = json.load(_file)
    _file.close()
    return _data


def _usage(cmd):
    print(
        "Usage: ",
        cmd,
        " [--comand|-c <cmd>] [--output|-o <ofile>] [--metadata|-m k[:type]=v] [--metadatafile <mdf>] <files>",
    )
    print("where:")
    print("  <files> are the files to read.")
    print("       File extension determines format. CSV is the text format")
    print("       from G4Analyais. For ROOT format, you need t specify the")
    print("       name of the file to be read in. Ex: file.root:h1")
    print(
        '       pickle format is supported (file should be created with command "save")'
    )
    print('  <cmd> is one of ("plot", "convert","save","genmd","list")')
    print('      "plot" (default) to plot the content of the file')
    print("      (requires matplotlib)")
    print('      "convert" creates an output file in JSON format suitable')
    print("      for FNALdb upload")
    print(
        '      "save" saves histograms in internal format to pickle file "histos.pkl"'
    )
    print('      "genmd" generates a metadata skeleton file as specified in <files>')
    print('      "list" shows content of ROOT File (TKey). Only for ROOT format.')
    print('  <ofile> is the output file name (default="output.json") for')
    print("      converted output for FNALdb")
    print(" <hn:k[:type]=v> is a key-value pair to add as metadata to FNALdb output")
    print("      hn is a regexp to assing the metadata to histogram based on names.")
    print(
        "      k is the key of the metadata, type (default INT) can be INT if the value has to"
    )
    print(
        "      interpreted as integer value of STR if it must be interpreted as string"
    )
    print("      or FLT if it is a floating point value")
    print(
        "      e.g. -m .*A:INT=1 means add to all objects the integer metadata 1 with key A"
    )
    print(
        " <mdf> is a json of pickle file containing the metadatada in a format of the type:"
    )
    print(
        '       { "regexpName" : { metadata } } where regexpName is a regular expression that'
    )
    print(
        "       matches a converted object name (the name being the ROOT TObject name or CSV full-filename)"
    )


def _handleCmdLineMetadata(metadata, mds):
    """
    Handle metadata passed via command line
    """
    for md in mds:
        # Global metadata, apply to all converted objects
        if md[0][-3:] not in ("INT", "FLT", "STR"):
            md[0] += ":INT"
        _hn, _k, _t = md[0].split(":")
        if _hn not in metadata:
            metadata[_hn] = {}
        if _t == "STR":
            metadata[_hn][_k] = md[1]
        elif _t == "INT":
            metadata[_hn][_k] = int(md[1])
        elif _t == "FLT":
            metadata[_hn][_k] = float(md[1])
        else:
            raise MetaDataError("ERROR: Unknown metadata type: %s" % _t)


##########################################
### Main
##########################################
if __name__ == "__main__":
    import sys
    import getopt

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hc:m:o:",
            ["command=", "metadata=", "output=", "metadatafile="],
        )
    except:
        _usage(sys.argv[0])
        exit(1)
    command = "plot"
    output = "output.json"
    mds = []
    metadata = {}
    for opt, arg in opts:
        if opt == "-h":
            _usage(sys.argv[0])
            exit(0)
        elif opt in ("-c", "--command"):
            command = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-m", "--metadata"):
            mds.append(arg.split("="))
        elif opt in ("--metadatafile"):
            metadata = _readMetaDataFile(arg)
    if command not in ("plot", "convert", "save", "genmd", "list"):
        _usage(sys.argv[0])
        exit(1)
    if len(args) == 0:
        _usage(sys.argv[0])
        exit(1)

    if command == "genmd":
        generate_metadata_file(args[0])
        exit(0)

    # Parsing commad line done
    _handleCmdLineMetadata(metadata, mds)

    histos = []
    # Loop on all histograms and parse them according to their type
    for _file in args:
        from os import path

        _ss = _file.split(":")
        _fn, _extension = path.splitext(_ss[0])
        _fname = _fn + _extension
        if _extension.lower() == ".pkl":
            import pickle

            _if = open(_fname)
            histos += pickle.load(_if)
        # ROOT file
        if _extension.lower() == ".root":
            # Check that ROOT name includes name of object to read,
            # if not assume it is all objects '*'
            if len(_ss) == 1:
                _file += ":*"
                _ss = _file.split(":")
            _hname = _ss[1]
            import ROOT

            tfile = ROOT.TFile(_fname)
            if not tfile.IsOpen():
                raise TFileNotOpen("Cannot open file %s" % _fname)
            # Navigate ROOT file structure retrieveing the full paths of all
            # TObjects in the file. For example:
            # [ 'tfilename.root/tdirectoryname/tobjectname']
            candidatesnames = []
            candidatesnames = _getObjsInTDir("", tfile, candidatesnames)
            # Will only read objects that match the requested pattern
            # Work with reg-exp but allow simpler '*' selection
            _orighname = _hname
            if "*" in _hname and ".*" not in _hname:
                _hname = _hname.replace("*", ".*")
            import re

            _mm = re.compile("^" + _hname + "$")
            _howmany = 0
            if command == "list":
                print("ROOT file content:")
            for cand in candidatesnames:
                if command == "list":
                    print("", cand, end=" ")
                cand = cand[
                    len(tfile.GetName()) + 1 :
                ]  # Remove prefix (ROOT file name)
                # print cand
                _ro = tfile.Get(cand)
                if command == "list":
                    print("(Type: %s)" % _ro.ClassName())
                if _mm.match(cand) and not command == "list":
                    try:
                        hfr = convert_from_root(_ro)
                        histos.append(hfr)
                        _howmany += 1
                    except ObjectNotSupported as e:
                        # Not a supported object type
                        print(
                            "WARNING: ROOT object %s of type %s not supported"
                            % (cand, _ro.ClassName())
                        )
                        pass
                    except Exception as e:
                        # I think this should never happen...
                        print(
                            "Cannot convert ROOT object:",
                            _ro.GetName(),
                            _ro.ClassName(),
                        )
                        raise e
            if _howmany == 0 and not command == "list":
                print(
                    "WARNING: no objecrs in",
                    tfile.GetName(),
                    "matching name",
                    _orighname,
                    "found",
                )
        # CSV (G4Analysis) type
        elif _extension.lower() == ".csv":
            lines = open(_file).readlines()
            histos.append(convert_from_csv(lines, _fname))
    if command == "list":
        exit(0)
    # Now execute action of all converted histograms
    if len(histos) == 0:
        raise EmptyInput("No objects found in input")
    if command == "plot":
        plot(histos)
    elif command == "save":
        import pickle

        _of = open("histos.pkl", "w")
        pickle.dump(histos, _of)
        _of.close()
    elif command == "convert":
        import re

        converted = []
        for h in histos:
            mds = {}
            for key in list(metadata.keys()):
                if type(metadata[key]) != dict:
                    msg = (
                        'ERROR: Metadata format error, expecting a dictionay for "%s \s"'
                        % key
                    )
                    msg += 'got "%s"' % str(metadata[key])
                    raise MetaDataError(msg)
                if re.match("^" + key.rstrip("$").lstrip("^") + "$", h["name"]):
                    mds = metadata[key]
                    # TODO: How to deal with multiple matches?
                    #       could do a merge of the metadata
                    break  # Note that we get the first metadata matching
            converted.append(convert_to_dossier(h, mds))
        results = {"ResultList": converted}
        #    [ convert_to_dossier(h,metadata) for h in histos ] }
        import json

        of = open(output, "w")
        json.dump(converted, of, indent=1)
    else:
        _usage(sys.argv[0])
        exit(1)
