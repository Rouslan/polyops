const POINT_HIT_CIRCLE_R = 5;

const state_classes = [null,"sweep","inverted","nested","tooriginal"];
const event_types = ['vforward','backward','forward','vbackward','calc_balance_intr','calc_intr_sample'];

function vGetX(a) {
    if(typeof a === 'number') return a;
    if(a instanceof Array) return a[0];
    return a.x;
}
function vGetY(a) {
    if(typeof a === 'number') return a;
    if(a instanceof Array) return a[1];
    return a.y;
}
const vreduce = (f,initial,vals) => [
    vals.reduce(((x1,b) => f(x1,vGetX(b))),vGetX(initial)),
    vals.reduce(((y1,b) => f(y1,vGetY(b))),vGetY(initial))];
const vadd = (a,...vals) => vreduce(((c1,c2) => c1 + c2),a,vals);
const vsub = (a,...vals) => vreduce(((c1,c2) => c1 - c2),a,vals);
const vmul = (a,...vals) => vreduce(((c1,c2) => c1 * c2),a,vals);
const vdiv = (a,...vals) => vreduce(((c1,c2) => c1 / c2),a,vals);
const vneg = a => [-vGetX(a),-vGetY(a)];
const clamp = (x,lo,hi) => x < lo ? lo : (x > hi ? hi : x);
const vapply = (f,...vals) => [f(...vals.map(vGetX)),f(...vals.map(vGetY))];
const vclamp = (a,lo,hi) => vapply(clamp,a,lo,hi);
function vsquare(a) {
    const x = vGetX(a);
    const y = vGetY(a);
    return x*x + y*y;
}
const getXY = a => [a.x,a.y];
const getWH = a => [a.width,a.height];
function setXY(a,p) {
    a.x = vGetX(p);
    a.y = vGetY(p);
    return a;
}
function setWH(a,p) {
    a.width = vGetX(p);
    a.height = vGetY(p);
    return a;
}
const rectCenter = r => [r.x + r.width/2,r.y + r.height/2];


function setSVGLength(attr,length,unit=SVGLength.SVG_LENGTHTYPE_PX) {
    attr.baseVal.newValueSpecifiedUnits(unit,length);
}

function viewExtent(points,padding) {
    let max_corner;
    let min_corner;

    if(points.length) {
        min_corner = max_corner = points[0].data;
        for(const p of points) {
            max_corner = vapply(Math.max,max_corner,p.data);
            min_corner = vapply(Math.min,min_corner,p.data);
        }
    } else {
        min_corner = max_corner = [0,0];
    }

    return [vadd(max_corner,padding),vsub(min_corner,padding)];
}

function connectClassToggle(checkElemId,targetElem,classname) {
    const checkElem = document.getElementById(checkElemId);
    checkElem.onchange = (e) => {
        if(e.target.checked) targetElem.classList.add(classname);
        else targetElem.classList.remove(classname);
    };
    if(checkElem.checked) targetElem.classList.add(classname);
}

function makeElem(etype,attrs={}) {
    const e = attrs.namespace === undefined ?
        document.createElement(etype) : document.createElementNS(attrs.namespace,etype);
    if(attrs.id !== undefined) e.id = attrs.id;
    if(attrs.classes !== undefined) {
        if(typeof attrs.classes === 'string') e.className = attrs.classes;
        else e.classList.add(...attrs.classes);
    }
    if(attrs.inner !== undefined) {
        if(Array.isArray(attrs.inner)) e.append(...attrs.inner);
        else e.append(attrs.inner);
    }
    return e;
}

function lineAtIndex(i) {
    return document.getElementById('p'+i);
}

class LineHighlighter {
    constructor(mainCName,lineCName) {
        this.mainCName = mainCName;
        this.lineCName = lineCName;
        this.mainElem = null;
        this.lines = null;
    }
    reset(mainElem=null,lines=[]) {
        if(this.mainElem !== null) {
            this.mainElem.classList.remove(this.mainCName);
            for(const line of this.lines)
                lineAtIndex(line).classList.remove(this.lineCName);
        }
        this.mainElem = mainElem;
        this.lines = lines;
        if(mainElem !== null) {
            mainElem.classList.add(this.mainCName);
            for(const line of lines)
                lineAtIndex(line).classList.add(this.lineCName);
        }
    }
}

function segmentStr(s,rev=false) {
    return s === null ? 'none' : `${s[rev ? 1 : 0]} - ${s[rev ? 0 : 1]}`;
}

export default function (inputCoords=null) {
    const svg = document.getElementsByTagName("svg")[0];
    const svgNS = 'http://www.w3.org/2000/svg';
    const socketO = document.getElementById('socketO');
    const socketOExtra = document.getElementById('socketOExtra');
    const statusBar = document.getElementById('statusbar');
    const zoomValue = document.getElementById('zoomValue');
    const buttons = [];
    let invViewTransform;
    let viewScale = 1;
    let viewOffset = [0,0];
    const originalViewBox = [null,null];
    let dragging = false;

    const hotElements = new LineHighlighter('hot','hot');

    const getInputPoint = i => [inputCoords[i*2],inputCoords[i*2+1]];
    const getInputLength = () => inputCoords.length / 2;

    function addLine(type,content,element='div') {
        const line = makeElem(element,{classes: type,inner: content});
        socketO.append(line);
        line.scrollIntoView();
        return line;
    }

    connectClassToggle('showrawtoggle',socketO,'showraw');
    connectClassToggle('showsweeptoggle',socketO,'showsweep');
    connectClassToggle('showvlabeltoggle',svg,'showvlabel');

    /* cancel out scaling applied to the SVG viewport for certain elements,
    using CSS */
    let rules = document.getElementsByTagName('style')[0].sheet.cssRules;
    let style = null;
    for(const r of rules) {
        if(r.selectorText == '.scaleinvariant') {
            style = r.style;
            break;
        }
    }
    if(style === null) throw Error('failed to find ".scaleinvariant" rule');

    function handleSVGSize() {
        const ctm = svg.getScreenCTM();
        if(ctm === null) return;
        invViewTransform = ctm.inverse();
        style.setProperty(
            'transform',
            `scale(${invViewTransform.a},${invViewTransform.d})`);
    }
    svg.addEventListener('resize',handleSVGSize);


    function clientToSVGCoords(x,y,w=1) {
        return getXY((new DOMPointReadOnly(x,y,0,w))
            .matrixTransform(invViewTransform));
    }

    function createSVGLength(length,unit=SVGLength.SVG_LENGTHTYPE_PX) {
        const r = svg.createSVGLength();
        r.newValueSpecifiedUnits(unit,length);
        return r;
    }

    function createSVGPolyLine(points) {
        const pl = document.createElementNS(svgNS,'polyline');
        for(const p of points) {
            pl.points.appendItem(setXY(svg.createSVGPoint(),p));
        }
        return pl;
    }

    function addSVGPolyLine(points,cssClass=null,id=null) {
        const p = createSVGPolyLine(points);
        if(cssClass !== null) p.classList.add(cssClass);
        if(id !== null) p.id = id;
        svg.appendChild(p);
    }

    function addPLabel(p,lines) {
        // make p into a unique reference
        p = [p[0],p[1]];

        const linestarts = lines.map(x => x[0]);

        const g = document.createElementNS(svgNS,'g');
        g.classList.add('scaleinvariant');
        g.style.setProperty('transform-origin',`${p[0]}px ${p[1]}px`);

        let e = document.createElementNS(svgNS,'text');
        e.x.baseVal.initialize(createSVGLength(p[0]));
        e.y.baseVal.initialize(createSVGLength(p[1]));
        e.classList.add('vlabel');
        e.appendChild(document.createTextNode(linestarts.join('/')));
        g.appendChild(e);

        e = document.createElementNS(svgNS,'circle');
        setSVGLength(e.cx,p[0]);
        setSVGLength(e.cy,p[1]);
        setSVGLength(e.r,POINT_HIT_CIRCLE_R);
        e.classList.add('hitarea');
        e.onmousemove = (e) => {
            let closest_d = null;
            let closest_elem = null;

            const mPoint = clientToSVGCoords(e.clientX,e.clientY);

            for(const elem of document.elementsFromPoint(e.clientX,e.clientY)) {
                if(elem.localName == 'circle') {
                    const d = vsquare(vsub(mPoint,[elem.cx.baseVal.value,elem.cy.baseVal.value]));
                    if(closest_d === null || d < closest_d) {
                        closest_d = d;
                        closest_elem = elem;
                    }
                }
            }

            if(closest_elem !== null) {
                const x = closest_elem.cx.baseVal.value;
                const y = closest_elem.cy.baseVal.value;
                const g_elem = closest_elem.parentElement;
                const text = g_elem.children[0].childNodes[0].data;
                statusBar.textContent = `point: ${x}, ${y} "${lines.map(x => x.join('\u2192')).join('/')}"`;
                if(g_elem !== hotElements.mainElem) hotElements.reset(g_elem,linestarts);
            }
        };
        e.onmouseleave = () => {
            statusBar.textContent = '\u00A0';
            hotElements.reset();
        };
        g.appendChild(e);

        svg.appendChild(g);
    }

    function updateZoom() {
        if(viewScale == 1) {
            setWH(setXY(svg.viewBox.baseVal,originalViewBox[0]),originalViewBox[1]);
            viewOffset = [0,0];
        } else {
            let scaled = vdiv(originalViewBox[1],viewScale);
            viewOffset = vclamp(viewOffset,0,vsub(originalViewBox[1],scaled));
            setWH(
                setXY(svg.viewBox.baseVal,vadd(originalViewBox[0],viewOffset)),
                scaled);
        }

        handleSVGSize();
    }

    function executeDrawing(points,currentPoint,indexedLineCount) {
        const padding = 10;
        const [maxCorner,minCorner] = viewExtent(points,padding);

        originalViewBox[0] = minCorner;
        originalViewBox[1] = vsub(maxCorner,minCorner);

        svg.textContent = null;
        updateZoom();

        if(currentPoint !== null) {
            addSVGPolyLine(
                [[currentPoint[0],minCorner[1]],[currentPoint[0],maxCorner[1]]],
                "sweepline");
        }

        if(inputCoords !== null) {
            const isize = getInputLength();
            let prev_p = getInputPoint(isize-1);
            for(let i=0; i<isize; ++i) {
                const p = getInputPoint(i);
                addSVGPolyLine([prev_p,p],"original");
                prev_p = p;
            }
        }

        points.forEach((p,i) => {
            let id = null;
            if(i < indexedLineCount) id = 'p'+i;
            addSVGPolyLine([p.data,points[p.next].data],state_classes[p.state],id);
        });

        const pointIndices = new Map();
        for(let i=0; i<indexedLineCount; ++i) {
            const p = points[i].data;
            const str = p.join(',');
            let indices = pointIndices.get(str);
            if(indices === undefined) {
                indices = {p:p,items:[]};
                pointIndices.set(str,indices);
            }
            indices.items.push([i,points[i].next]);
        }

        for(const {p,items} of pointIndices.values()) {
            addPLabel(p,items);
        }
    }

    function setSvgZoom(z,focus=null) {
        if(focus === null) focus = rectCenter(svg.viewBox.baseVal);

        if(z < 1) z = 1;
        else if(z > 16) z = 16;
        else z = Math.round((z+Number.EPSILON)*10000) / 10000;

        if(z == viewScale) return;

        /* make it so that the SVG coordinates "focus" stays in the same spot
        on the screen, before and after zooming */
        viewOffset = vsub(
            vadd(
                vmul(vsub(getXY(svg.viewBox.baseVal),focus),viewScale/z),
                focus),
            originalViewBox[0]);

        viewScale = z;
        z = (z*100).toFixed(2);
        if(z.endsWith('.00')) z = z.slice(0,-3);
        else if(z.endsWith('0')) z = z.slice(0,-1);
        zoomValue.value = z;

        updateZoom();
    }

    zoomValue.onchange = e => {
        let z = parseFloat(zoomValue.value);
        if(isNaN(z)) zoomValue.value = svg.currentScale*100;
        else setSvgZoom(z/100);
    };

    document.getElementById('bZoomIn').onclick = () => { setSvgZoom(viewScale*1.1); };
    document.getElementById('bZoomOut').onclick = () => { setSvgZoom(viewScale/1.1); };

    svg.onpointerdown = e => {
        if(!e.isPrimary) return;
        dragging = true;
        svg.setPointerCapture(e.pointerId);
    };
    svg.onpointerup = e => {
        if(!e.isPrimary) return;
        svg.releasePointerCapture(e.pointerId);
        dragging = false;
    };
    svg.onlostpointercapture = e => {
        if(!e.isPrimary) return;
        dragging = false;
    };
    svg.onpointermove = e => {
        if(!(e.isPrimary && dragging)) return;
        viewOffset = vsub(viewOffset,clientToSVGCoords(e.movementX,e.movementY,0));
        updateZoom();
    };
    svg.onwheel = e => {
        e.preventDefault();
        let d = -e.deltaY;
        if(d == 0) return;

        /* There doesn't seem to be an easy way to get scroll wheel ticks. 48 is
        the value produced by a single tick on my system. */
        if(e.deltaMode == WheelEvent.DOM_DELTA_PIXEL) d /= 48;

        setSvgZoom(
            viewScale*Math.pow(1.1,d),
            clientToSVGCoords(e.clientX,e.clientY));
    };

    const selectedLB = new LineHighlighter('active','lbhit');
    function addLineBalanceInfo(pi,hits,balance) {
        const line = addLine(
            'lbinfo',
            `line balance of point ${pi}: ${balance}  hits: ${hits.join(',')}`);
        line.tabIndex = 0;
        const toggle = e => {
            if(selectedLB.mainElem === e.currentTarget) selectedLB.reset();
            else selectedLB.reset(e.currentTarget,hits);
        };
        line.onclick = toggle;
        line.onkeydown = e => {
            if(e.code == 'Space' || e.code == 'Enter') toggle(e);
        };
    }

    let selectedMsg = null;
    function forwardBackwardEvent(forward,segment,before,after,sweep,events) {
        const sweepItems = [];
        for(const s of sweep) sweepItems.push(makeElem('li',{inner: segmentStr(s)}));
        const msg = addLine('eventMessage',[
            makeElem('div',{inner: `event: ${forward ? 'FORWARD' : 'BACKWARD'} ${segmentStr(segment,!forward)}`}),
            makeElem('div',{classes: 'extrainfo',inner: [
                makeElem('div',{inner: `before: ${segmentStr(before)}`}),
                makeElem('div',{inner: `after: ${segmentStr(after)}`}),
                makeElem('div',{classes: 'sweep',inner: [
                    makeElem('div',{inner: 'Sweep items:'}),
                    makeElem('ul',{inner: sweepItems})
                ]})
            ]})
        ]);
        msg.tabIndex = 0;
        const activate = e => {
            if(selectedMsg !== e.currentTarget) {
                if(selectedMsg !== null) selectedMsg.classList.remove('active');
                selectedMsg = e.currentTarget;
                selectedMsg.classList.add('active');
                socketOExtra.textContent = null;
                for(const e of events) {
                    const entry = makeElem('div',{inner: `${event_types[e.type]}: ${e.ab[0]} - ${e.ab[1]}`});
                    if(e.deleted) entry.classList.add('deleted');
                    socketOExtra.append(entry);
                }
            }
        };
        msg.onclick = activate;
        msg.onkeydown = e => {
            if(e.code == 'Space' || e.code == 'Enter') activate(e);
        };
    }

    try {
        var ws = new WebSocket("ws://" + location.host);
        ws.onclose = e => {
            for(const b of buttons) b.disabled = true;
            if(e.wasClean) addLine('message',"connection closed: " + e.reason);
            else addLine('error',"connection error: " + e.reason);
        };
        ws.onmessage = e => {
            addLine('raw',e.data);
            const msg = JSON.parse(e.data);
            switch(msg.command) {
            case "draw":
                executeDrawing(msg.points,msg.currentPoint,msg.indexedLineCount);
                break;
            case "console":
                addLine('message',msg.text,'pre');
                break;
            case "linebalance":
                addLineBalanceInfo(msg.point,msg.hits,msg.balance);
                break;
            case "originalpoints":
                inputCoords = msg.points;
                break;
            case 'event':
                switch(msg.type) {
                case "forward":
                case "backward":
                    forwardBackwardEvent(msg.type == 'forward',msg.segment,msg.before,msg.after,msg.sweep,msg.events);
                    break;
                }
                break;
            default:
                addLine('error',`unrecognised command: "${msg.command}"`);
            }
        };
        ws.onopen = e => {
            for(const b of buttons) b.disabled = false;
        };

        let b = document.getElementById('bContinue');
        buttons.push(b);
        b.onclick = e => { ws.send('continue'); };
    } catch(e) {
        addLine('error',"error: " + e.message);
    }
}
