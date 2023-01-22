const svgNS = 'http://www.w3.org/2000/svg';
const svgStyle = `svg .hitarea {
    fill: none;
}
.hot > text { fill: red; }
polyline.hot, svg polygon.hot { stroke-width: 2px; }
polyline, polygon {
    stroke: black;
    vector-effect: non-scaling-stroke;
}
.vlabel {
    text-anchor: middle;
    dominant-baseline: middle;
    font-family: monospace;
    font-size: 12px;
    display: none;
}
.showvlabel .vlabel {
    display: unset;
}
.sweep { stroke: #ff9999; }
.inverted, svg .sweepline { stroke: green; }
.nested { stroke: blue; }
.tooriginal { stroke: grey; }
.original { stroke: purple; }
polyline .lbhit { stroke: orange; }`;

function downloadFile(source,anchor) {
    const doc = new Document();
    const root = doc.importNode(source,true);
    doc.appendChild(root);
    const style = doc.createElementNS(svgNS,'style');
    style.textContent = svgStyle;
    root.prepend(style);
    const url = URL.createObjectURL(new Blob(
        [(new XMLSerializer()).serializeToString(doc)],
        {type: 'image/svg+xml'}));

    try {
        anchor.href = url;
        const event = document.createEvent('Event');
        event.initEvent('click',true,true);
        anchor.dispatchEvent(event);
    } finally {
        /*URL.revokeObjectURL(url);*/
    }
}

export default function (source,button,anchor) {
    button.onclick = e => {
        downloadFile(source,anchor);
    };
}
