// Stand-in DOM for running the page's script under JavaScriptCore
// (osascript -l JavaScript): records table renders and canvas drawing calls.
var LOG=[]; var handlers={}; var ELS={}; var ctxCalls=0;
function el(id){ return {id:id, value:(id==="f-status"?"all":id==="f-dv"?"0":id==="f-src"?"all":id==="f-spec"?"all":""), textContent:"", hidden:false, className:"", _html:"",
  set innerHTML(v){this._html=v; LOG.push(id+" html "+v.length);}, get innerHTML(){return this._html;},
  appendChild:function(){}, addEventListener:function(t,f){handlers[id+":"+t]=f;}, querySelector:function(){return null;}, dataset:{} }; }
function ctx(){ return new Proxy({}, {get:function(t,k){ if(["setLineDash","scale","beginPath","moveTo","lineTo","stroke","fill","closePath","fillText"].indexOf(k)>=0) return function(){ctxCalls++;}; return t[k];}, set:function(t,k,v){t[k]=v;return true;}}); }
var document={
  getElementById:function(id){ if(id==="data") return {textContent:DATA_TXT}; if(id==="spec") return {textContent:SPEC_TXT}; return ELS[id]||(ELS[id]=el(id)); },
  createElement:function(){ return {}; },
  querySelectorAll:function(sel){ if(sel.indexOf("canvas")===0){ var m=sel.match(/data-row="(\d+)"/); var rr=m?m[1]:null; if(!rr||!SPEC_OBJ[rr]) return []; var out=[]; for(var i=0;i<SPEC_OBJ[rr].panels.length;i++) out.push({dataset:{p:String(i)}, clientWidth:270, clientHeight:150, getContext:function(){return ctx();}}); return out; } return []; },
  addEventListener:function(t,f){ handlers["doc:"+t]=f; }, documentElement:{}, activeElement:null };
var window={devicePixelRatio:2};
function getComputedStyle(){ return {getPropertyValue:function(){return "#123456";}}; }
function setTimeout(){ return 1; } function clearTimeout(){}
