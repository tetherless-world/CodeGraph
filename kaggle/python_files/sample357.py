#!/usr/bin/env python
# coding: utf-8

# # Interactive D3.js Visualisations in Kaggle Kernels 
# 
# D3.js is a JavaScript library for manipulating documents based on data. D3 helps you bring data to life using HTML, SVG, and CSS. D3’s emphasis on web standards gives you the full capabilities of modern browsers without tying yourself to a proprietary framework, combining powerful visualization components and a data-driven approach to DOM manipulation. I am using the dataset of stackoverflow survey 2018 to create these visualizations. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.core.display import display, HTML, Javascript
from string import Template
import pandas as pd
import numpy as np
import json, random
import IPython.display
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/survey_results_public.csv')


#############################
##  
t = df['Country'].value_counts().to_frame().reset_index()
strr = "id,value,value1\nproject,\n"
num = 1
cnt = 1
sizes =[9000,7500,6000,5000,4000,2500,2200,1900,1800,1860]
for j,x in t.iterrows():
    val = x['Country']
    strr += "project." +str(num)+"."+ x['index'].replace('"','').replace("'","").split(",")[0] + "," + str(val) + "," + str(x['Country']) + "\n"
    if cnt % 2 == 0:
        num += 1
    cnt += 1
    if cnt == 100:
        break
fout = open("flare1.csv", "w")
fout.write(strr)

html_p1 = """<!DOCTYPE html><svg id="one" width="760" height="760" font-family="sans-serif" font-size="10" text-anchor="middle"></svg>"""
js_p1 = """require.config({paths: {d3: "https://d3js.org/d3.v4.min"}});
require(["d3"], function(d3) {var svg=d3.select("#one"),width=+svg.attr("width"),height=+svg.attr("height"),format=d3.format(",d"),color=d3.scaleOrdinal(d3.schemeCategory20c);console.log(color);var pack=d3.pack().size([width,height]).padding(1.5);d3.csv("flare1.csv",function(t){if(t.value=+t.value,t.value)return t},function(t,e){if(t)throw t;var n=d3.hierarchy({children:e}).sum(function(t){return t.value}).each(function(t){if(e=t.data.id){var e,n=e.lastIndexOf(".");t.id=e,t.package=e.slice(0,n),t.class=e.slice(n+1)}}),a=(d3.select("body").append("div").style("position","absolute").style("z-index","10").style("visibility","hidden").text("a"),svg.selectAll(".node").data(pack(n).leaves()).enter().append("g").attr("class","node").attr("transform",function(t){return"translate("+t.x+","+t.y+")"}));a.append("circle").attr("id",function(t){return t.id}).attr("r",function(t){return t.r}).style("fill",function(t){return color(t.package)}),a.append("clipPath").attr("id",function(t){return"clip-"+t.id}).append("use").attr("xlink:href",function(t){return"#"+t.id}),a.append("svg:title").text(function(t){return t.value}),a.append("text").attr("clip-path",function(t){return"url(#clip-"+t.id+")"}).selectAll("tspan").data(function(t){return t.class.split(/(?=[A-Z][^A-Z])/g)}).enter().append("tspan").attr("x",0).attr("y",function(t,e,n){return 13+10*(e-n.length/2-.5)}).text(function(t){return t})});});
"""

##############################
##  
from collections import Counter 
def parse(col):
    bigtxt = ";".join(df[col].dropna())
    wrds = bigtxt.split(";")
    wrds = Counter(wrds).most_common()
    resp = {'name' : col, "children" : [], "size":""}
    for wrd in wrds:
        doc = {'name' : wrd[0], "size": wrd[1]}
        resp['children'].append(doc)
    return resp
results = {'name' : 'flare', "children" : [], "size":""}
languages = parse('LanguageWorkedWith')
results['children'].append(languages)
frameworks = parse('FrameworkWorkedWith')
results['children'].append(frameworks)
frameworks = parse('DatabaseWorkedWith')
results['children'].append(frameworks)
frameworks = parse('VersionControl')
results['children'].append(frameworks)
frameworks = parse('OperatingSystem')
results['children'].append(frameworks)
frameworks = parse('IDE')
results['children'].append(frameworks)
frameworks = parse('Methodology')
results['children'].append(frameworks)
frameworks = parse('PlatformWorkedWith')
results['children'].append(frameworks)
frameworks = parse('CommunicationTools')
results['children'].append(frameworks)
with open('output.json', 'w') as outfile:  
    json.dump(results, outfile)

htmlt1 = """<!DOCTYPE html><meta charset="utf-8"><style>.node {cursor: pointer;}.node:hover {stroke: #000;stroke-width: 1.5px;}.node--leaf {fill: white;}
.label {font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;text-anchor: middle;text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;}
.label,.node--root,.node--leaf {pointer-events: none;}</style><svg id="two" width="760" height="760"></svg>
"""
js_t1="""require(["d3"], function(d3) {
var svg = d3.select("#two"),
    margin = 20,
    diameter = +svg.attr("width"),
    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")"),
    color = d3.scaleSequential(d3.interpolateViridis).domain([-2, 2]),
    pack = d3.pack().size([diameter - margin, diameter - margin]).padding(2);
d3.json("output.json", function(t, n) {
if (t) throw t;
var r, e = n = d3.hierarchy(n).sum(function(t) {
        return t.size
    }).sort(function(t, n) {
        return n.value - t.value
    }),
    a = pack(n).descendants(),
    i = g.selectAll("circle").data(a).enter().append("circle").attr("class", function(t) {
        return t.parent ? t.children ? "node" : "node node--leaf" : "node node--root"
    }).style("fill", function(t) {
        return t.children ? color(t.depth) : null
    }).on("click", function(t) {
        e !== t && (l(t), d3.event.stopPropagation())
    }),
    o = (g.selectAll("text").data(a).enter().append("text").attr("class", "label").style("fill-opacity", function(t) {
        return t.parent === n ? 1 : 0
    }).style("display", function(t) {
        return t.parent === n ? "inline" : "none"
    }).text(function(t) {
        return t.data.name + ": " + t.data.size
    }), g.selectAll("circle,text"));

function l(t) {
    e = t, d3.transition().duration(d3.event.altKey ? 7500 : 750).tween("zoom", function(t) {
        var n = d3.interpolateZoom(r, [e.x, e.y, 2 * e.r + margin]);
        return function(t) {
            c(n(t))
        }
    }).selectAll("text").filter(function(t) {
        return t.parent === e || "inline" === this.style.display
    }).style("fill-opacity", function(t) {
        return t.parent === e ? 1 : 0
    }).on("start", function(t) {
        t.parent === e && (this.style.display = "inline")
    }).on("end", function(t) {
        t.parent !== e && (this.style.display = "none")
    })
}

function c(n) {
    var e = diameter / n[2];
    r = n, o.attr("transform", function(t) {
        return "translate(" + (t.x - n[0]) * e + "," + (t.y - n[1]) * e + ")"
    }), i.attr("r", function(t) {
        return t.r * e
    })
}
svg.style("background", color(-1)).on("click", function() {
    l(n)
}), c([n.x, n.y, 2 * n.r + margin])
});
});"""
##################
# third 

##

##############################
# fourth 
doc = {"name": "Characteristics", "color": "#ffae00", "percent": "", "value": "", "size": 25, "children": []}

def getsize(s):
    if s > 80:
        return 30
    elif s > 65:
        return 20
    elif s > 45:
        return 15
    elif s > 35:
        return 12
    elif s > 20:
        return 10 
    else:
        return 5
def vcs(col):
    vc = df[col].value_counts()
    keys = vc.index
    vals = vc.values 
    
    ddoc = {"name": col, "color": "#be5eff", "percent": "", "value": "", "size": 25, "children": []}
    for i,x in enumerate(keys):
        percent = round(100 * float(vals[i]) / sum(vals), 2)
        size = getsize(percent)
        if x[0] == "N":
            collr = "#fc5858"
        elif x[0] == "Y":
            collr = "#a2ff77"
        else:
            collr = "#aeccfc"
        doc = {"name": x+" ("+str(percent)+"%)", "color": collr, "percent": str(percent), "value": str(vals[i]), "size": size, "children": []}
        ddoc['children'].append(doc)
    return ddoc

# Coding Backgrounds
doc['children'].append(vcs('Hobby'))
doc['children'].append(vcs('OpenSource'))
doc['children'].append(vcs('Student'))
doc['children'].append(vcs('AdBlocker'))
doc['children'].append(vcs('Dependents'))
doc['children'].append(vcs('MilitaryUS'))

html_d1 = """<!DOCTYPE html><style>.node text {font: 12px sans-serif;}.link {fill: none;stroke: #ccc;stroke-width: 2px;}</style><svg id="four" width="760" height="900" font-family="sans-serif" font-size="10" text-anchor="middle"></svg>"""
js_d1="""
require(["d3"], function(d3) {
var treeData = """ +json.dumps(doc) + """
var root, margin = {
        top: 20,
        right: 90,
        bottom: 120,
        left: 90
    },
    width = 960 - margin.left - margin.right,
    height = 660,
    svg = d3.select("#four").attr("width", width + margin.right + margin.left).attr("height", height + margin.top + margin.bottom).append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")"),
    i = 0,
    duration = 750,
    treemap = d3.tree().size([height, width]);

function collapse(t) {
    t.children && (t._children = t.children, t._children.forEach(collapse), t.children = null)
}

function update(n) {
    var t = treemap(root),
        r = t.descendants(),
        e = t.descendants().slice(1);
    r.forEach(function(t) {
        t.y = 180 * t.depth
    });
    var a = svg.selectAll("g.node").data(r, function(t) {
            return t.id || (t.id = ++i)
        }),
        o = a.enter().append("g").attr("class", "node").attr("transform", function(t) {
            return "translate(" + n.y0 + "," + n.x0 + ")"
        }).on("click", function(t) {
            t.children ? (t._children = t.children, t.children = null) : (t.children = t._children, t._children = null);
            update(t)
        });
    o.append("circle").attr("class", "node").attr("r", function(t) {
        return t.data.size
    }).style("fill", function(t) { return t.data.color;
    }), o.append("text").attr("dy", ".35em").attr("x", function( t) {
        return t.children || t._children ? -13 : 13
    }).attr("text-anchor", function(t) {
        return t.children || t._children ? "end" : "start"
    }).text(function(t) {
        return t.data.name
    });
    var c = o.merge(a);
    c.transition().duration(duration).attr("transform", function(t) {
        return "translate(" + t.y + "," + t.x + ")"
    }), c.select("circle.node").attr("r", function(t) {
        return t.data.size
    }).style("fill", function(t) {
        return t.data.color
    }).attr("cursor", "pointer");
    var l = a.exit().transition().duration(duration).attr("transform", function(t) {
        return "translate(" + n.y + "," + n.x + ")"
    }).remove();
    l.select("circle").attr("r", function(t) {
        return t.data.size
    }), l.select("text").style("fill-opacity", 1e-6);
    var d = svg.selectAll("path.link").data(e, function(t) {
        return t.id
    });
    console.log(), d.enter().insert("path", "g").attr("class", "link").attr("d", function(t) {
        var r = {
            x: n.x0,
            y: n.y0
        };
        return u(r, r)
    }).merge(d).transition().duration(duration).attr("d", function(t) {
        return u(t, t.parent)
    });
    d.exit().transition().duration(duration).attr("d", function(t) {
        var r = {
            x: n.x,
            y: n.y
        };
        return u(r, r)
    }).remove();

    function u(t, r) {
        var n = "M" + t.y + "," + t.x + "C" + (t.y + r.y) / 2 + "," + t.x + " " + (t.y + r.y) / 2 + "," + r.x + " " + r.y + "," + r.x;
        return console.log(n), n
    }
    r.forEach(function(t) {
        t.x0 = t.x, t.y0 = t.y
    })
}(root = d3.hierarchy(treeData, function(t) {
    return t.children
})).x0 = height / 2, root.y0 = 0, root.children.forEach(collapse), update(root);
});
"""




###################################################################################################3
def parse_demo(col):
    bigtxt = ";".join(df[col].dropna())
    wrds = bigtxt.split(";")
    wrds = Counter(wrds).most_common()
    return wrds 

strr = "id,value,color\nAudience Demographics,\n"
demographics = ['Gender', 'Age', 'SexualOrientation', 'RaceEthnicity', 'EducationParents', 'Dependents']
#demographics = ['Gender', 'Age', 'SexualOrientation', 'RaceEthnicity', 'EducationParents']
colors = ['#5b9aff', '#ff77bd', '#82ff8a', '#9b9493', '#5b9aff', '#ff77bd', '#82ff8a', '#9b9493']
for i,col in enumerate(demographics):
    strr += "Audience Demographics." + col + ",\n"
    
    response = parse_demo(col)
    total = sum([x[1] for x in response])
    for term in response:
        cent = float(term[1])*100 / total
        strr += "Audience Demographics." + col +"."+ term[0].split("(")[0].replace(",","") +","+ str(cent) + ","+colors[i]+"\n"

fout = open("tomdata.csv", "w")

fout.write(strr)



html2 =""" <style>
.link {
        fill: none;
        stroke: #555;
        stroke-opacity: 0.4;
        stroke-width: 1px;
    }
    text {
        font-family: "Arial Black", Gadget, sans-serif;
        fill: black;
        font-weight: bold;
        font-size: 14px
    }

    .xAxis .tick text{
        fill: black;
    }
    .grid .tick line{
        stroke: grey;
        stroke-dasharray: 5, 10;
        opacity: 0.7;
    }
    .grid path{
        stroke-width: 0;
    }

    .node1 circle {
        fill: #999;
    }
    .node1--internal circle {
        fill: #555;
    }
    .node1--internal text {
        font-size: 16px;
        text-shadow: 0 2px 0 #fff, 0 -2px 0 #fff, 2px 0 0 #fff, -2px 0 0 #fff;
    }
    .node1--leaf text {
        fill: white;
    }
    .ballG text {
        fill: white;
    }

    .shadow {
        -webkit-filter: drop-shadow( -1.5px -1.5px 1.5px #000 );
        filter: drop-shadow( -1.5px -1.5px 1.5px #000 );
    }</style>
    <body>
    <br><br>
    <svg id="five" width="900" height="1200"></svg>
    <br><br><br>
</body>
"""


js2 = """
 
 require(["d3"], function(d3) {
  
    var svg1 = d3.select("#five"),
            width = +svg1.attr("width"),
            height = +svg1.attr("height"),
            g1 = svg1.append("g").attr("transform", "translate(20,10)");       // move right 20px.

    var xScale =  d3.scaleLinear()
            .domain([0,100])
            .range([0, 400]);

    var xAxis = d3.axisTop()
            .scale(xScale);

    // Setting up a way to handle the data
    var tree1 = d3.cluster()                 // This D3 API method setup the Dendrogram datum position.
            .size([height, width - 550])    // Total width - bar chart width = Dendrogram chart width
            .separation(function separate(a, b) {
                return a.parent == b.parent            // 2 levels tree1 grouping for category
                || a.parent.parent == b.parent
                || a.parent == b.parent.parent ? 0.4 : 0.8;
            });

    var stratify = d3.stratify()            // This D3 API method gives cvs file flat data array dimensions.
            .parentId(function(d) { return d.id.substring(0, d.id.lastIndexOf(".")); });

    d3.csv("tomdata.csv", row, function(error, data) {
        if (error) throw error;

        var root1 = stratify(data);
        tree1(root1);

        // Draw every datum a line connecting to its parent.
        var link = g1.selectAll(".link")
                .data(root1.descendants().slice(1))
                .enter().append("path")
                .attr("class", "link")
                .attr("d", function(d) {
                    return "M" + d.y + "," + d.x
                            + "C" + (d.parent.y + 100) + "," + d.x
                            + " " + (d.parent.y + 100) + "," + d.parent.x
                            + " " + d.parent.y + "," + d.parent.x;
                }).style("stroke", function(d) { 
                        if (d.data.id=="Audience Demographics.Dependents"){
                        return "white";
                        }
                        if (d.data.id=="Audience Demographics.MilitaryUS"){
                        return "white";
                        }
        });

        // Setup position for every datum; Applying different css classes to parents and leafs.
        var node1 = g1.selectAll(".node1")
                .data(root1.descendants())
                .enter().append("g")
                .attr("class", function(d) { return "node1" + (d.children ? " node1--internal" : " node1--leaf"); })
                .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

        // Draw every datum a small circle.
        node1.append("circle")
                .attr("r", 4);

        // Setup G for every leaf datum.
        var leafNode1G = g1.selectAll(".node1--leaf")
                .append("g")
                .attr("class", "node1--leaf-g")
                .attr("transform", "translate(" + 8 + "," + -13 + ")");

        leafNode1G.append("rect")
                .attr("class","shadow")
                .style("fill", function (d) {return d.data.color;})
                .attr("width", 2)
                .attr("height", 30)
                .attr("rx", 2)
                .attr("ry", 2)
                .transition()
                    .duration(800)
                    .attr("width", function (d) {return xScale(d.data.value);});

        leafNode1G.append("text")
                .attr("dy", 19.5)
                .attr("x", 8)
                .style("text-anchor", "start")
                .style("fill", "#222")
                .text(function (d) {
                    return d.data.id.substring(d.data.id.lastIndexOf(".") + 1);
                });

        // Write down text for every parent datum
        var internalNode1 = g1.selectAll(".node1--internal");
        internalNode1.append("text")
                .attr("y", -10)
                .style("text-anchor", "middle")
                .text(function (d) {
                    return d.data.id.substring(d.data.id.lastIndexOf(".") + 1);
                });

        // Attach axis on top of the first leaf datum.
        var firstEndNode1 = g1.select(".node1--leaf");
            firstEndNode1.insert("g")
                    .attr("class","xAxis")
                    .attr("transform", "translate(" + 7 + "," + -14 + ")")
                    .call(xAxis);

            // tick mark for x-axis
            firstEndNode1.insert("g")
                    .attr("class", "grid")
                    .attr("transform", "translate(7," + (height - 15) + ")")
                    .call(d3.axisBottom()
                            .scale(xScale)
                            .ticks(5)
                            .tickSize(-height, 0, 0)
                            .tickFormat("")
                    );

        // Emphasize the y-axis baseline.
        svg1.selectAll(".grid").select("line")
                .style("stroke-dasharray","20,1")
                .style("stroke","black");


    });

    function row(d) {
        return {
            id: d.id,
            value: +d.value,
            color: d.color
        };
    }
    });"""
#####################33

def get_size(t):
    if t > 20000:
        return 28
    elif t > 15000:
        return 25
    elif t > 12000:
        return 23
    elif t > 11000:
        return 21
    elif t > 10000:
        return 18
    elif t > 9000:
        return 16
    elif t > 8000:
        return 13
    elif t > 6000:
        return 10
    else:
        return 5

nodes=[]# nodes = [{'id' : 'Questions', 'group' : 0, 'size' : 18}]
links = []
# links.append({"source": 'Questions', "target": "AssessJob", "value": 1})
assessJob = ['AssessJob1', 'AssessJob2', 'AssessJob3', 'AssessJob4', 'AssessJob5', 'AssessJob6', 'AssessJob7', 'AssessJob8', 'AssessJob9', 'AssessJob10']
nodes.append({'id' : 'AssessJob', 'group' : 0, 'size' : 15})
for i, col in enumerate(assessJob):
    nodes.append({'id' : col, 'group' : i+1, "size": 12})
    t = df[col].dropna().value_counts()
    x = list(t.index) 
    y = list(t.values)
    links.append({"source": 'AssessJob', "target": col, "value": 1})
    
    for j, each in enumerate(x):
        idd = col + " Rated: " + str(each).replace(".0","") + " (" + str(y[j]) + ")"
        nodes.append({'id' : idd, 'group' : i+1, "size" : get_size(y[j])}) 
        links.append({"source": col, "target": idd, "value": 1})

doc = {'nodes' : nodes, 'links' : links}
with open("fd.json", "w") as fout:
    fout.write(json.dumps(doc))
########################################################3

img = """iVBORw0KGgoAAAANSUhEUgAAAsYAAAK9CAIAAACo7z3gAAAAinpUWHRSYXcgcHJvZmlsZSB0eXBlIGV4aWYAAHjaVY7BDQMhDAT/riIlLLZZcDkJuZPSQcqPERehm4e9WqExcnw/pzwmBSpeW2eQSDw89JmhY2FAUZS5cy6ubSWT7lpMV2D0Bt8P/er/VGPn2bw1Vg4OTbsepsaceUimFfMbsSWDWBq79683bnr5AVCyLLXeKf2hAAAKBmlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4KPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iWE1QIENvcmUgNC40LjAtRXhpdjIiPgogPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4KICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgeG1sbnM6ZXhpZj0iaHR0cDovL25zLmFkb2JlLmNvbS9leGlmLzEuMC8iCiAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyIKICAgZXhpZjpQaXhlbFhEaW1lbnNpb249IjcxMCIKICAgZXhpZjpQaXhlbFlEaW1lbnNpb249IjcwMSIKICAgdGlmZjpJbWFnZVdpZHRoPSI3MTAiCiAgIHRpZmY6SW1hZ2VIZWlnaHQ9IjcwMSIKICAgdGlmZjpPcmllbnRhdGlvbj0iMSIvPgogPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg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"""
import base64
import io
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

i = base64.b64decode(img)
i = io.BytesIO(i)
i = mpimg.imread(i, format='JPG')

plt.figure(figsize=(12,8))
plt.imshow(i, interpolation='nearest')
plt.axis('off')
plt.savefig("test.png", bbox_inches='tight')
plt.show()

# ## 1. Simple Bubble Plot 
# ### Top Countries of StackOverFlow Audience 
# 
# Lets plot the country names of the stackover flow audience in a bubble chart. The size of every bubble repersents the number of respondents from that country.  

# In[ ]:


h = display(HTML(html_p1))
j = IPython.display.Javascript(js_p1)
IPython.display.display_javascript(j)

# > **Hover on the bubbles** to view the count of respondents from each country. 
# 
# ## 2. Zoomable Hierarichial Bubble Plot  
# ### Key Characteristics of Stackoverflow Audience 
# 
# Lets visualize the Key Skills, Frameworks, Databases, VersonControl, OperatingSystems, IDE, Methodology, and EthicsChoice popular among the audience using Interactive Zoomable Bubble Chart. This size of bubble repersents the number of respondents with respect to the particular skill.  

# In[ ]:


html5="""<!DOCTYPE html>
<html>
  <head>
  <style>
#main {
  float: left;
  width: 750px;
}

#sidebar {
  float: right;
  width: 100px;
}

#sequence {
  width: 600px;
  height: 70px;
}

#legend {
  padding: 10px 0 0 3px;
}

#sequence text, #legend text {
  font-weight: 600;
  fill: #fff;
}

#chart {
  position: relative;
}

#chart path {
  stroke: #fff;
}

#explanation {
  position: absolute;
  top: 330px;
  left: 405px;
  width: 120px;
  text-align: center;
  color: #666;
  z-index: -1;
}

#percentage {
  font-size: 2.5em;
}</style>
    <meta charset="utf-8">
    <title>Sequences sunburst</title>
  </head>
  <body>
    <div id="main">
      <div id="sequence"></div>
      <svg id="chart">
        <div id="explanation" style="visibility: hidden;">
          <span id="percentage"></span><br/>
          Respondents
        </div>
      </svg>
    </div>
  </body>
</html>"""

vis = """DevTypes-Technical-Backend,53300
DevTypes-Technical-Fullstack,44353
DevTypes-Technical-Frontend,34822
DevTypes-Technical-Mobile,18804
DevTypes-Technical-Applications,15807
DevTypes-Technical-EmbeddedApps,4819
DevTypes-Misc-Student,15732
DevTypes-Systems-Database,13216
DevTypes-Design-Designer,12019
DevTypes-Systems-SystemAdmin,10375
DevTypes-Systems-DevOps,9549
DevTypes-DataScience-Analyst,7559
DevTypes-DataScience-DataScientist,7088
DevTypes-Testing-QA,6194
DevTypes-Management-Engineering,5256
DevTypes-Design-Graphics,4642
DevTypes-Management-Product,4316
DevTypes-Misc-Researcher,3641
DevTypes-Management-Csuite,3491
DevTypes-Management-MarketingSales,1122"""
fout = open("visit-sequences.csv","w")
fout.write(vis)

h = display(HTML(htmlt1))
j = IPython.display.Javascript(js_t1)
IPython.display.display_javascript(j)

# ## The above graph is interactive, Click on the bubbles to zoom in and zoom out. 
# 
# ## 3. Collapsable Tree 
# ### Traits of Stack Overflow Audience
# 
# Lets visualize the Key Traits of the Stack Overflow Audience 

# In[ ]:


js7m="""require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });
 
 require(["d3"], function(d3) {// Dimensions of sunburst.
 
 


var svg = d3.select("#fd"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()

    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(120).strength(1))
    .force("charge", d3.forceManyBody().strength(-155))
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("fd.json", function(error, graph) {
  if (error) throw error;

  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

// Define the div for the tooltip
var div = d3.select("body").append("div")	
    .attr("class", "tooltip")				
    .style("opacity", 0);

  var node = svg.append("g")
      .attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
      .attr("r", function(d) {return d.size})
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)).on("mouseover", function(d) {		
            div.transition()		
                .duration(200)		
                .style("opacity", .9);		
            div	.html(d.id )
                .style("left", (d3.event.pageX) + "px")		
                .style("top", (d3.event.pageY - 28) + "px");	
            })					
        .on("mouseout", function(d) {		
            div.transition()		
                .duration(500)		
                .style("opacity", 0);	
        });
          
    
// node.append("title")
  //  .text(function(d) { return d.id; });

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);
      

  simulation.force("link")
      .links(graph.links);

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }
});

function dragstarted(d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}
 });
"""

h = display(HTML(html_d1))
j = IPython.display.Javascript(js_d1)
IPython.display.display_javascript(j)

# ## The above graph is **interactive**, Click on the Nodes to Interact and obtain More Info
# 
# 
# ## 4. Dendogram Tree with Bars 
# ### Demographics of Stack OverFlow Audience
# 
# Lets visualize and understand the demographics of the stackoverflow audience / respondents. 

# In[ ]:


html7="""<!DOCTYPE html>
<meta charset="utf-8">
<style>

.links line {
  stroke: #999;
  stroke-opacity: 0.8;
}
.node text {
  pointer-events: none;
  font: 10px sans-serif;
}

.tooldiv {
    display: inline-block;
    width: 120px;
    background-color: white;
    color: #000;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;
    z-index: 1;
}
.nodes circle {
  stroke: #fff;
  stroke-width: 1.5px;
}

div.tooltip {	
    position: absolute;			
    text-align: center;			
    width: 100px;					
    height: 40px;					
    padding: 2px;				
    font: 12px sans-serif;		
    background: lightsteelblue;	
    border: 0px;		
    border-radius: 8px;			
    pointer-events: none;			
}

</style>
<svg id="fd" width="760" height="760"></svg>"""

h = display(HTML(html2))
j = IPython.display.Javascript(js2)
IPython.display.display_javascript(j)

# ## 5. SunBurst Chart 
# ### Developer Types of the Audience 
# 
# Lets visualize the designations and roles of the stack overflow audience. We know that stack overflow is the place for developers and technical queries, however a large majority of different designations visit this website. I have categoried the different designations into few categories such as technical, management, DataScience, Systems etc. The following visualization is called sun burst visualization. 

# In[ ]:


js7="""require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });
 
 require(["d3"], function(d3) {// Dimensions of sunburst.
 
 


var svg = d3.select("#fd"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()
    // fix the link distance, charge and the center layout  
    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(120).strength(1))
    .force("charge", d3.forceManyBody().strength(-155))
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("fd.json", function(error, graph) {
  if (error) throw error;

  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

// Define the div for the tooltip
var div = d3.select("body").append("div")	
    .attr("class", "tooltip")				
    .style("opacity", 0);

  var node = svg.append("g")
      .attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
      .attr("r", function(d) {return d.size})
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)).on("mouseover", function(d) {		
            div.transition()		
                .duration(200)		
                .style("opacity", .9);		
            div	.html(d.id )
                .style("left", (d3.event.pageX) + "px")		
                .style("top", (d3.event.pageY - 28) + "px");	
            })					
        .on("mouseout", function(d) {		
            div.transition()		
                .duration(500)		
                .style("opacity", 0);	
        });
          
    
// node.append("title")
  //  .text(function(d) { return d.id; });

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);
      

  simulation.force("link")
      .links(graph.links);

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }
});

function dragstarted(d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}
 });
"""

js5 = """
 
 require(["d3"], function(d3) {// Dimensions of sunburst. reduce to lower dimensions  
var width = 750;
var height = 600;
var radius = Math.min(width, height) / 2;

// Breadcrumb dimensions: width, height, spacing, width of tip/tail.
var b = {
  w: 150, h: 30, s: 3, t: 10
};

// Mapping of step names to colors.
// Mapping of step names to colors. manually created this 
var colors = {
  "DevTypes" : "#bc99c6",
  "Technical":"#f4c542",
  "Misc": "#5687d1",
  "Design": "#7b615c",
  "Management": "#de783b",
  "Testing": "#6ab975",
  "DataScience": "#a173d1",
  "Systems": "#bbbbb",

  "Backend":"#ef4078",
  "Fullstack": "#5687d1",
  "Frontend": "#7b615c",
  "Mobile": "#de783b",
  "Applications": "#6ab975",
  "EmbeddedApps": "#a173d1",
  "Student": "#bbbbb",

  "Database":"#f4c542",
  "Designer": "#5687d1",
  "SystemAdmin": "#7b615c",
  "DevOps": "#de783b",
  "Analyst": "#6ab975",
  "QA": "#a173d1",
  "DataScientist": "#bbbbb",

  "Engineering":"#f4c542",
  "Product": "#5687d1",
  "Graphics": "#7b615c",
  "Researcher": "#de783b",
  "Analyst": "#6ab975",
  "Csuite": "#a173d1",
  "MarketingSales": "#bbbbb"
};

// Total size of all segments; we set this later, after loading the data.
var totalSize = 0; 

var vis = d3.select("#chart")
    .attr("width", width)
    .attr("height", height)
    .append("svg:g")
    .attr("id", "container")
    .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

var partition = d3.partition()
    .size([2 * Math.PI, radius * radius]);

var arc = d3.arc()
    .startAngle(function(d) { return d.x0; })
    .endAngle(function(d) { return d.x1; })
    .innerRadius(function(d) { return Math.sqrt(d.y0); })
    .outerRadius(function(d) { return Math.sqrt(d.y1); });

d3.text("visit-sequences.csv", function(text) {
  var csv = d3.csvParseRows(text);
  var json = buildHierarchy(csv);
  createVisualization(json);
});

// Main function to draw and set up the visualization, once we have the data.
function createVisualization(json) {

  // Basic setup of page elements.
  initializeBreadcrumbTrail();
  //drawLegend();
  //d3.select("#togglelegend").on("click", toggleLegend);

  // Bounding circle underneath the sunburst, to make it easier to detect
  // when the mouse leaves the parent g.
  vis.append("svg:circle")
      .attr("r", radius)
      .style("opacity", 0);

  // Turn the data into a d3 hierarchy and calculate the sums.
  var root = d3.hierarchy(json)
      .sum(function(d) { return d.size; })
      .sort(function(a, b) { return b.value - a.value; });
  
  // For efficiency, filter nodes to keep only those large enough to see.
  var nodes = partition(root).descendants()
      .filter(function(d) {
          return (d.x1 - d.x0 > 0.005); // 0.005 radians = 0.29 degrees
      });

  var path = vis.data([json]).selectAll("path")
      .data(nodes)
      .enter().append("svg:path")
      .attr("display", function(d) { return d.depth ? null : "none"; })
      .attr("d", arc)
      .attr("fill-rule", "evenodd")
      .style("fill", function(d) { return colors[d.data.name]; })
      .style("opacity", 1)
      .on("mouseover", mouseover);

  // Add the mouseleave handler to the bounding circle.
  d3.select("#container").on("mouseleave", mouseleave);

  // Get total size of the tree = value of root node from partition.
  totalSize = path.datum().value;
 };

// Fade all but the current sequence, and show it in the breadcrumb trail.
function mouseover(d) {

  var percentage = (100 * d.value / totalSize).toPrecision(3);
  var percentageString = percentage + "%";
  if (percentage < 0.1) {
    percentageString = "< 0.1%";
  }

  d3.select("#percentage")
      .text(percentageString);

  d3.select("#explanation")
      .style("visibility", "");

  var sequenceArray = d.ancestors().reverse();
  sequenceArray.shift(); // remove root node from the array
  updateBreadcrumbs(sequenceArray, percentageString);

  // Fade all the segments.
  d3.selectAll("path")
      .style("opacity", 0.3);

  // Then highlight only those that are an ancestor of the current segment.
  vis.selectAll("path")
      .filter(function(node) {
                return (sequenceArray.indexOf(node) >= 0);
              })
      .style("opacity", 1);
}

// Restore everything to full opacity when moving off the visualization.
function mouseleave(d) {

  // Hide the breadcrumb trail
  d3.select("#trail")
      .style("visibility", "hidden");

  // Deactivate all segments during transition.
  d3.selectAll("path").on("mouseover", null);

  // Transition each segment to full opacity and then reactivate it.
  d3.selectAll("path")
      .transition()
      .duration(1000)
      .style("opacity", 1)
      .on("end", function() {
              d3.select(this).on("mouseover", mouseover);
            });

  d3.select("#explanation")
      .style("visibility", "hidden");
}

function initializeBreadcrumbTrail() {
  // Add the svg area.
  var trail = d3.select("#sequence").append("svg:svg")
      .attr("width", width)
      .attr("height", 50)
      .attr("id", "trail");
// Avoid the conflict with the other graphs
// Removing this part - Add the label at the end, for the percentage.
 // trail.append("svg:text")
   // .attr("id", "endlabel")
    //.style("fill", "#000");
}

// Generate a string that describes the points of a breadcrumb polygon.
function breadcrumbPoints(d, i) {
  var points = [];
  points.push("0,0");
  points.push(b.w + ",0");
  points.push(b.w + b.t + "," + (b.h / 2));
  points.push(b.w + "," + b.h);
  points.push("0," + b.h);
  if (i > 0) { // Leftmost breadcrumb; don't include 6th vertex.
    points.push(b.t + "," + (b.h / 2));
  }
  return points.join(" ");
}

// Update the breadcrumb trail to show the current sequence and percentage.
function updateBreadcrumbs(nodeArray, percentageString) {

  // Data join; key function combines name and depth (= position in sequence).
  var trail = d3.select("#trail")
      .selectAll("g")
      .data(nodeArray, function(d) { return d.data.name + d.depth; });

  // Remove exiting nodes.
  trail.exit().remove();

  // Add breadcrumb and label for entering nodes.
  var entering = trail.enter().append("svg:g");

  entering.append("svg:polygon")
      .attr("points", breadcrumbPoints)
      .style("fill", function(d) { return colors[d.data.name]; });

  entering.append("svg:text")
      .attr("x", (b.w + b.t) / 2)
      .attr("y", b.h / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .text(function(d) { return d.data.name; });

  // Merge enter and update selections; set position for all nodes.
  entering.merge(trail).attr("transform", function(d, i) {
    return "translate(" + i * (b.w + b.s) + ", 0)";
  });

  // Now move and update the percentage at the end.
  d3.select("#trail").select("#endlabel")
      .attr("x", (nodeArray.length + 0.5) * (b.w + b.s))
      .attr("y", b.h / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .text(percentageString);

  // Make the breadcrumb trail visible, if it's hidden.
  d3.select("#trail")
      .style("visibility", "");

}


function buildHierarchy(csv) {
  var root = {"name": "root", "children": []};
  for (var i = 0; i < csv.length; i++) {
    var sequence = csv[i][0];
    var size = +csv[i][1];
    if (isNaN(size)) { // e.g. if this is a header row
      continue;
    }
    var parts = sequence.split("-");
    var currentNode = root;
    for (var j = 0; j < parts.length; j++) {
      var children = currentNode["children"];
      var nodeName = parts[j];
      var childNode;
      if (j + 1 < parts.length) {
   // Not yet at the end of the sequence; move down the tree.
  var foundChild = false;
  for (var k = 0; k < children.length; k++) {
    if (children[k]["name"] == nodeName) {
      childNode = children[k];
      foundChild = true;
      break;
    }
  }
  if (!foundChild) {
    childNode = {"name": nodeName, "children": []};
    children.push(childNode);
  }
  currentNode = childNode;
      } else {
  childNode = {"name": nodeName, "size": size};
  children.push(childNode);
      }
    }
  }
  return root;
};

 });"""

h = display(HTML(html5))
j = IPython.display.Javascript(js5)
IPython.display.display_javascript(j)

# ## Above chart is also interactive, hover from the inner circles to the outer circles to view the info
# 
# 
# Tip -> Start hovering from the innermost circle to the outermost ones in order to view the breakdown and distribution
# 
# 
# ## 6. Forced Directed Network 
# ### Job Assessment Responses
# 
# The respondents were asked several questions about job assessment and their responses were captured from 1 to 10. Lets visualize the responses of all the questions in one chart using force directed graph. The questions were the following. Though the choices are not known, but the visualization can help to understand were the choices showed too much variation or there were one or two popular choices, and which choice got the highest responses.    
# 

# In[ ]:


html7_1="""<!DOCTYPE html>
<meta charset="utf-8">
<style>

.links line {
  stroke: #999;
  stroke-opacity: 0.8;
}
.node text {
  pointer-events: none;
  font: 10px sans-serif;
}

.tooldiv {
    display: inline-block;
    width: 120px;
    background-color: white;
    color: #000;
    text-align: center;
    padding: 5px 0;
    border-radius: 6px;
    z-index: 1;
}
.nodes circle {
  stroke: #fff;
  stroke-width: 1.5px;
}

div.tooltip {	
    position: absolute;			
    text-align: center;			
    width: 100px;					
    height: 40px;					
    padding: 2px;				
    font: 12px sans-serif;		
    background: lightsteelblue;	
    border: 0px;		
    border-radius: 8px;			
    pointer-events: none;			
}

</style>
<svg id="fd" width="760" height="760"></svg>"""

h = display(HTML(html7))
j = IPython.display.Javascript(js7)
IPython.display.display_javascript(j)

# ## Above chart is also interactive, hover on the nodes to view the information
# 
# ## 7. Zoomable Icicle Plot 
# ### Interest of Audience for Hypothetical Tools
# 
# Lets visulize the interest of audience using an Icicle Plot.

# In[ ]:


from collections import Counter 



colnames = """A peer mentoring system
A private area for people new to programming
A programming-oriented blog platform
An employer or job review system
An area for Q&A related to career growth"""
colnames = [ccoo for ccoo in colnames.split("\n")]


def parse(col):
    bigtxt = ";".join(df[col].dropna().astype(str))
    wrds = bigtxt.split(";")
    wrds = Counter(wrds).most_common()
        
    col = colnames[int(col.replace("HypotheticalTools","")) - 1]
#     resp = {'name' : col.replace("Hypothetical","").replace("s"," "), "children" : [], "size":""}
    resp = {'name' : col, "children" : [], "size":""}
    for wrd in wrds:
        doc = {'name' : wrd[0] + " ("+ str(wrd[1]) + " Respondents)", "size": wrd[1]}
        resp['children'].append(doc)
    return resp


results = {'name' : '', "children" : [], "size":""}
languages = parse('HypotheticalTools1')
results['children'].append(languages)
frameworks = parse('HypotheticalTools2')
results['children'].append(frameworks)
frameworks = parse('HypotheticalTools3')
results['children'].append(frameworks)
frameworks = parse('HypotheticalTools4')
results['children'].append(frameworks)
frameworks = parse('HypotheticalTools5')
results['children'].append(frameworks)

with open('output_me.json', 'w') as outfile:  
    json.dump(results, outfile)


d = json.loads(open('output_me.json').read())

res = {"Interest of Respondents on Hypothetical Tools on StackOverflow" : {}}
for each in d['children']:
    if each['name'] not in res["Interest of Respondents on Hypothetical Tools on StackOverflow"]:
        res['Interest of Respondents on Hypothetical Tools on StackOverflow'][each['name']] = {}
        
    for x in each['children']:
        res['Interest of Respondents on Hypothetical Tools on StackOverflow'][each['name']][x['name']] = x['size']

fout = open("master.json", "w")
fout.write(json.dumps(res))

jjsxx = """
 require(["d3"], function(d3) {// Dimensions of sunburst.

var width = 960,
    height = 500;

var x = d3.scaleLinear()
    .range([0, width]);

var y = d3.scaleLinear()
    .range([0, height]);

var color = d3.scaleOrdinal(d3.schemeCategory20c);

var vist = d3.select('#icicle')
    .attr("width", width)
    .attr("height", height)

var partitiont = d3.partition()
    .size([width, height])
    .padding(0)
    .round(true);
	
// Breadcrumb dimensions: width, height, spacing, width of tip/tail.
var b = {
  w: 150, h: 30, s: 3, t: 10
};

var rect = vist.selectAll("rect");
var fo = vist.selectAll("foreignObject");
var totalSize=0;

d3.json("master.json", function(error, roott) {
  if (error) throw error;

  roott = d3.hierarchy(d3.entries(roott)[0], function(d) {
      return d3.entries(d.value)
    })
    .sum(function(d) { return d.value })
    .sort(function(a, b) { return b.value - a.value; });

  partitiont(roott);
  
	  var sequenceArray = roott.ancestors().reverse();

  rect = rect
      .data(roott.descendants())
    .enter().append("rect")
      .attr("x", function(d) { return d.x0; })
      .attr("y", function(d) { return d.y0; })
      .attr("width", function(d) { return d.x1 - d.x0; })
      .attr("height", function(d) { return d.y1 - d.y0; })
      //.attr("fill", function(d) { return color((d.children ? d : d.parent).data.key); })
      .attr("fill", function(d) { 
          
          var strng = d.data.key
        var lst = strng.substr(strng.length - 1)
         if (lst == ")"){
             if (strng.charAt(0) == "S"){
                 return "#f285e1"
             }
             else if (strng.charAt(0) == "N"){
                 return "#7cefef"
             }
              else if (strng.charAt(0) == "V"){
                 return "#dfef7c"
             }
              else if (strng.charAt(0) == "E"){
                 return "#efa67c"
             }
             else{
                 return "#a4b1f9"
             }
         }
         else if (strng.charAt(0) == "A"){
             return "#fc7486"
         }
         else {
             return "#8af797"
         }
      
      })
      .on("click", clicked);
	  
	fo = fo
		.data(roott.descendants())
		.enter().append("foreignObject")
      .attr("x", function(d) { return d.x0; })
      .attr("y", function(d) { return d.y0; })
      .attr("width", function(d) { return d.x1 - d.x0; })
      .attr("height", function(d) { return d.y1 - d.y0; })
     .style("cursor", "pointer")
     .text(function(d) { 
     
        var strng = d.data.key
        var lst = strng.substr(strng.length - 1)
         if (lst == ")"){
         return ""
         }
         else{
         return d.data.key
         }

    
     
     })
     .on("click", clicked);
	 
	 //get total size from rect
	totalSize = rect.node().__data__.value;
});

function clicked(d) {
	
	x.domain([d.x0, d.x1]);
	y.domain([d.y0, height]).range([d.depth ? 20 : 0, height]);

	rect.transition()
      .duration(750)
      .attr("x", function(d) { return x(d.x0); })
      .attr("y", function(d) { return y(d.y0); })
      .attr("width", function(d) { return x(d.x1) - x(d.x0); })
      .attr("height", function(d) { return y(d.y1) - y(d.y0); });
	  
	  fo.transition()
        .duration(750)
      .attr("x", function(d) { return x(d.x0); })
      .attr("y", function(d) { return y(d.y0); })
      .attr("width", function(d) { return x(d.x1-d.x0); })
      .attr("height", function(d) { return y(d.y1-d.y0); });
      
    if (d.depth == 0){
         fo.
         text(function (d){return d.data.key}).style("opacity", function(d){
            
        var strng = d.data.key
        var lst = strng.substr(strng.length - 1)
         if (lst == ")"){
             return 0
         }
         else{
             return 1
         }
         
         })
     }
     else{
         fo.text(function(d) { d.data.key }).style("opacity", 1)
     }
     
     fo.text(function(d) { return d.data.key})
	  

	  var sequenceArray = d.ancestors().reverse();
}




});"""

hhtmlxx = """<!DOCTYPE html>
<meta charset="utf-8">
<style>

rect {
  stroke: #fff;
}


</style>
<body>
<div id="breadcrumb"></div>
<svg id="icicle"></svg>
</body>
"""

#### SOME TEXT IS DISTURBED, So Run it later
h = display(HTML(hhtmlxx))
j = IPython.display.Javascript(jjsxx)
IPython.display.display_javascript(j)

# ## Above chart is also interactive, click on the cells to view the information
# 
# ## 8. Categorical HeatMap 
# ### Job Assessment Response + Job Assessment Benefits Response
# 
# In the previous chart - 6, I visualized the Job Assessment response of the survey respondents, Let us visualize the data + job assessment benefits response in a heatmap. 

# In[ ]:


colors = ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58"]
def getccol(percent):
    if percent > 14:
        col = colors[-1]
    elif percent > 12:
        col = colors[-2]
    elif percent > 10:
        col = colors[-3]
    elif percent > 9:
        col = colors[-4]
    elif percent > 8:
        col = colors[-5]
    elif percent > 7:
        col = colors[-6]
    elif percent > 6:
        col = colors[-7]
    elif percent > 5:
        col = colors[-8]
    else:
        col = colors[-9]
    return col

fout= open("data_h1.tsv","w")
fout.write("day\thour\tx\ty\tvalue\tcolor\n")
cols = [i for i in range(1,11)]

days = []
times = []
for p,each in enumerate(cols):
    col = "AssessJob"+str(each)
    days.append(col)
    t = df[col].value_counts()
    mapp = {}
    for l,each in enumerate(t.index):
        mapp[each] = t.values[l]

    for key, value in mapp.items():
        percent = int(100*float(value)/sum(list(mapp.values())))
        colr = getccol(percent)
        
        row = [p+1, int(key), col, value, percent, colr]
        row = "\t".join([str(b) for b in row])
        fout.write(row + "\n")
        
times = [i for i in range(1,11)]

fout= open("data_h2.tsv","w")
fout.write("day\thour\tx\ty\tvalue\tcolor\n")
cols = [i for i in range(1,12)]

days2 = []
times2 = [i for i in range(1,12)]
for p,each in enumerate(cols):
    
    col = "AssessBenefits"+str(each)
    days2.append(col)
    t = df[col].value_counts()

    mapp = {}
    for l,each in enumerate(t.index):
        mapp[each] = t.values[l]

    for key, value in mapp.items():
        percent = int(100*float(value)/sum(list(mapp.values())))
        colr = getccol(percent)
        
        row = [p+1, int(key), col, value, percent, colr]
        row = "\t".join([str(b) for b in row])
        fout.write(row + "\n")

hhtml1 = """<!DOCTYPE html>
<meta charset="utf-8">
<html>
  <head>
    <style>
      rect.bordered {
        stroke: #fff;
        stroke-width:5px;
      }

      text.mono {
        font-size: 9pt;
        font-family: Consolas, courier;
        fill: #222;
      }

      text.axis-workweek {
        fill: #000;
      }

      text.axis-worktime {
        fill: #000;
      }
      div.tooltip {	
        position: absolute;			
        text-align: center;			
        width: 40px;					
        height: 20px;					
        padding: 2px;				
        font: 12px sans-serif;		
        background: lightsteelblue;	
        border: 0px;		
        border-radius: 8px;			
        pointer-events: none;			
    }
    </style>
  </head>
    <div id="chheat1"></div>
    <div id="chheat2"></div>
    </div>"""



jjs1 = """ require(["d3"], function(d3) {// Dimensions of sunburst.
 
     const margin = { top: 50, right: 0, bottom: 100, left: 150 },
          width = 960 - margin.left - margin.right,
          height = 500 - margin.top - margin.bottom,
          gridSize = Math.floor(width / 24),
          legendElementWidth = gridSize*2,
          buckets = 9,
          colors = ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58"], // alternatively colorbrewer.YlGnBu[9]
          days = """ + json.dumps(days) + """, 

          times = """+ json.dumps(times) + """;
          
    var div = d3.select("body").append("div")   
    .attr("class", "tooltip")               
    .style("opacity", 0);
          
    const svg = d3.select("#chheat1").append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g")
          .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


      var dayLabels = svg.selectAll(".dayLabel")
          .data(days)
          .enter().append("text")
            .text(function (d) { return d; })
            .attr("x", 0)
            .attr("y", (d, i) => i * gridSize)
            .style("text-anchor", "end")
            .attr("transform", "translate(-6," + gridSize / 1.5 + ")")
            .attr("class", (d, i) => ((i >= 0 && i <= 4) ? "dayLabel mono axis" : "dayLabel mono axis"));


      const timeLabels = svg.selectAll(".timeLabel")
          .data(times)
          .enter().append("text")
            .text((d) => d)
            .attr("x", (d, i) => i * gridSize)
            .attr("y", 0)
            .style("text-anchor", "middle")
            .attr("transform", "translate(" + gridSize / 2 + ", -6)")
            .attr("class", (d, i) => ((i >= 7 && i <= 16) ? "timeLabel mono axis" : "timeLabel mono axis"));

      const type = (d) => {
        return {
          day: +d.day,
          hour: +d.hour,
          value: +d.value,
          x : +d.x,
          y : +d.y
        };
      };


      const heatmapChart = function(tsvFile) {
        
        d3.tsv(tsvFile, type, (error, data) => {
          
          const colorScale = d3.scaleQuantile()
            .domain([0, buckets - 1, d3.max(data, (d) => d.value)])
            .range(colors);

            const cards = svg.selectAll(".hour")
              .data(data, (d) => d.day+':'+d.hour);

          var cardsEnter = cards.enter().append("rect")
              .attr("x", (d) => (d.hour - 1) * gridSize)
              .attr("y", (d) => (d.day - 1) * gridSize)
              .attr("rx", 4)
              .attr("ry", 4)
              .attr("class", "hour bordered")
              .attr("width", gridSize)
              .attr("height", gridSize)
              .style("fill", colors[0]);
          
        cardsEnter.on("mouseover", function(d) {      
            div.transition()        
                .duration(200)      
                .style("opacity", .9);      
            div.html(d.value + " %")
                .style("left", (d3.event.pageX) + "px")     
                .style("top", (d3.event.pageY - 28) + "px");    
            })                  
        .on("mouseout", function(d) {       
            div.transition()        
                .duration(500)      
                .style("opacity", 0);   
        });


           cardsEnter.merge(cards)
              .transition()
              .duration(1000)
              .style("fill", (d) => colorScale(d.value));
          cards.exit().remove();
                
        });
        }

    heatmapChart("data_h1.tsv");
 });
 """


jjs12 = """ require(["d3"], function(d3) {// Dimensions of sunburst.
 
     const margin = { top: 50, right: 0, bottom: 100, left: 150 },
          width = 960 - margin.left - margin.right,
          height = 500 - margin.top - margin.bottom,
          gridSize = Math.floor(width / 24),
          legendElementWidth = gridSize*2,
          buckets = 9,
          colors = ["#ffffd9","#edf8b1","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58"], // alternatively colorbrewer.YlGnBu[9]
          days2 = """ + json.dumps(days2) + """, 
          times = """+ json.dumps(times2) + """;
          
    var div = d3.select("body").append("div")   
    .attr("class", "tooltip")               
    .style("opacity", 0);
          
    const svg2 = d3.select("#chheat2").append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g")
          .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


      svg2.selectAll(".dayLabel2")
          .data(days2)
          .enter().append("text")
            .text(function (d) { return d; })
            .attr("x", 0)
            .attr("y", (d, i) => i * gridSize)
            .style("text-anchor", "end")
            .attr("transform", "translate(-6," + gridSize / 1.5 + ")")
            .attr("class", (d, i) => ((i >= 0 && i <= 4) ? "dayLabel2 mono axis" : "dayLabel2 mono axis"));


      const timeLabels = svg2.selectAll(".timeLabel")
          .data(times)
          .enter().append("text")
            .text((d) => d)
            .attr("x", (d, i) => i * gridSize)
            .attr("y", 0)
            .style("text-anchor", "middle")
            .attr("transform", "translate(" + gridSize / 2 + ", -6)")
            .attr("class", (d, i) => ((i >= 7 && i <= 16) ? "timeLabel mono axis" : "timeLabel mono axis"));

      const type = (d) => {
        return {
          day: +d.day,
          hour: +d.hour,
          value: +d.value,
          x : +d.x,
          y : +d.y
        };
      };


      const heatmapChart = function(tsvFile) {
        
        d3.tsv(tsvFile, type, (error, data) => {
          
          const colorScale = d3.scaleQuantile()
            .domain([0, buckets - 1, d3.max(data, (d) => d.value)])
            .range(colors);

            const cards = svg2.selectAll(".hour")
              .data(data, (d) => d.day+':'+d.hour);

          var cardsEnter = cards.enter().append("rect")
              .attr("x", (d) => (d.hour - 1) * gridSize)
              .attr("y", (d) => (d.day - 1) * gridSize)
              .attr("rx", 4)
              .attr("ry", 4)
              .attr("class", "hour bordered")
              .attr("width", gridSize)
              .attr("height", gridSize)
              .style("fill", colors[0]);
          
        cardsEnter.on("mouseover", function(d) {      
            div.transition()        
                .duration(200)      
                .style("opacity", .9);      
            div.html(d.value + " %")
                .style("left", (d3.event.pageX) + "px")     
                .style("top", (d3.event.pageY - 28) + "px");    
            })                  
        .on("mouseout", function(d) {       
            div.transition()        
                .duration(500)      
                .style("opacity", 0);   
        });


           cardsEnter.merge(cards)
              .transition()
              .duration(1000)
              .style("fill", (d) => colorScale(d.value));
          cards.exit().remove();
                
        });
        }

    heatmapChart("data_h2.tsv");
 });
 """

## plot both the charts  
h = display(HTML(hhtml1))
j = IPython.display.Javascript(jjs1)
IPython.display.display_javascript(j)
j = IPython.display.Javascript(jjs12)
IPython.display.display_javascript(j)

# ## Above chart is also interactive, hover on the squares to view the information

# ## Thanks for viewing the kernel. If you liked it, Please Upvote :)
