// Validation only: original ggmlc loader, graph executor and Laya preprocessing.
// This executable is never linked to or invoked by the Goish application.
#include "engine.h"
#include "presets.h"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace laya;
template<class T> JsonValue array(const std::vector<T>& v){auto a=JsonValue::array();for(auto x:v)a.arr.push_back(JsonValue::number(x));return a;}
int main(int argc,char**argv){
 if(argc==2 && std::string(argv[1])=="presets"){
 auto root=JsonValue::object();for(auto&p:all_presets()){auto j=JsonValue::object();j.set("state",p.default_state);j.set("questions",questions_to_json(p.questions));j.set("state_key",JsonValue::string(p.state_key));root.set(p.name,j);}std::cout<<json_dumps(root)<<"\n";return 0;}
 if(argc<3)return 1;
 auto graph=ggmlc::ModelLoader::load_from_file(argv[1]);
 DecisionRecipe recipe;recipe.load_from_graph(graph,argv[1]);
 ggmlc::pipeline::BPETokenizer tok;tok.init_from_gguf_file(argv[1]);
 if(std::string(argv[2])=="tokenize"){std::string line;while(std::getline(std::cin,line)){auto j=JsonParser::parse_string(line);std::cout<<json_dumps(array(tok.encode(j.s,0,false,false)))<<"\n";}return 0;}
 ggmlc::ModelExecutor executor(graph,"cpu");
 std::ifstream f(argv[2]);std::string text((std::istreambuf_iterator<char>(f)),{});auto req=JsonParser::parse_string(text);
 auto qs=questions_from_json(*req.get("questions"));auto state=serialize_state(*req.get("state"));auto out=JsonValue::object();
 for(auto&q:qs){
 auto enc=build_sequence(tok,recipe,state,q);int s=std::max(recipe.min_seq,(int)enc.ids.size());
 std::vector<int32_t>ids,mpos,qt,dec;std::vector<float>att,mm;
 pad_encoded_batch({enc},recipe,1,s,ids,att,mpos,mm,qt,dec);
 std::unordered_map<std::string,int64_t> env;
 auto&it=graph.tensors.at(graph.inputs[0]);
 env[graph.symbol_table[it.ne[0]->val]]=s;env[graph.symbol_table[it.ne[1]->val]]=1;
 executor.prepare(env,true);
 executor.set_input(graph.inputs[0],ids.data(),ids.size()*4);
 executor.set_input(graph.inputs[1],att.data(),att.size()*4);
 executor.set_input(graph.inputs[2],mpos.data(),mpos.size()*4);
 executor.set_input(graph.inputs[3],mm.data(),mm.size()*4);
 executor.set_input(graph.inputs[4],qt.data(),qt.size()*4);
 executor.run(4);
 auto lp=(const float*)executor.get_output_data(graph.outputs[0]);auto ap=(const float*)executor.get_output_data(graph.outputs[1]);
 auto a=JsonValue::object();a.set("input_ids",array(enc.ids));a.set("markers",array(enc.markers));a.set("logits",array(std::vector<float>(lp,lp+enc.markers.size())));a.set("act_logits",array(std::vector<float>(ap,ap+2)));out.set(q.id,a);
 }
 std::cout<<json_dumps(out)<<"\n";
}
