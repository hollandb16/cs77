#include "scene.h"
#include "intersect.h"
#include "montecarlo.h"
#include "animation.h"
#include "tesselation.h"

#include <thread>
using std::thread;

// modify the following line to disable/enable parallel execution of the pathtracer
bool parallel_pathtrace = true;
bool blur = false;
float blurSize = .1; //higher is more diffuse
float russian_threshold = 0; //Set to zero to ignore russian roulette

image3f pathtrace(Scene* scene, bool multithread);
void pathtrace(Scene* scene, image3f* image, RngImage* rngs, int offset_row, int skip_row, bool verbose);

// lookup texture value
vec3f lookup_scaled_texture(vec3f value, image3f* texture, vec2f uv, bool tile = false) {
    if(not texture) return value;
    auto i = int(uv.x*texture->width());
    auto j = int(uv.y*texture->height());

    auto s = uv.x * texture->width() - i;
    auto t = uv.y * texture->height() - j;

    int i_prime = i+1;
    int j_prime = j+1;

    if(tile){
        i%=texture->width();
        if(i < 0){ i += texture->width();}

        i_prime%=texture->width();
        if(i_prime < 0){ i_prime += texture->width(); }

        j%=texture->height();
        if(j < 0){ j += texture->height(); }

        j_prime%=texture->height();
        if(j_prime < 0){ j_prime += texture->height(); }
    }
    else{
        i = clamp(i, 0, texture->width()-1);
        i_prime = clamp(i_prime, 0, texture->width()-1);
        j = clamp(j, 0, texture->height()-1);
        j_prime = clamp(j_prime, 0, texture->height()-1);
    }

    return value * (texture->at(i,j)*(1-s)*(1-t) +
                    texture->at(i,j_prime)*(1-s)*t +
                    texture->at(i_prime,j)*s*(1-t) +
                    texture->at(i_prime,j_prime)*s*t);
}

// compute the brdf
vec3f eval_brdf(vec3f kd, vec3f ks, float n, vec3f v, vec3f l, vec3f norm, bool microfacet) {
    if (not microfacet) {
        auto h = normalize(v+l);
        return kd/pif + ks*(n+8)/(8*pif) * pow(max(0.0f,dot(norm,h)),n);
    } else {
        //put_your_code_here("Implement microfacet brdf");
        auto h = normalize(v+l);

        auto d = ((n+2)/(2.0*pif)) * pow(max(0.0f,dot(norm,h)), n);

        auto f = ks + (one3f - ks)*pow((1-dot(h, norm)), 5);

        auto numer1 = 2*dot(h,norm)*dot(v,norm);
        auto numer2 = 2*dot(h,norm)*dot(l,norm);
        auto g = min(numer1/dot(v,h), numer2/dot(l,h));

        g = min(1.0, g);

        auto p = (d*g*f)/(4*dot(l,norm)*dot(v,norm));

       return p;
       //return one3f;  // <- placeholder

    }
}

// evaluate the environment map
vec3f eval_env(vec3f ke, image3f* ke_txt, vec3f dir) {

    float u = atan2(dir.x, dir.z)/(2.0f*pif);

    float v = 1.0f - acos(dir.y)/pif;

    return lookup_scaled_texture(ke, ke_txt, vec2f(u,v), true);
}

// compute the color corresponing to a ray by pathtrace
vec3f pathtrace_ray(Scene* scene, ray3f ray, Rng* rng, int depth) {
    
    //Russian Roulette
    if(rng->next_float() < russian_threshold) return zero3f;

    // get scene intersection
    auto intersection = intersect(scene,ray);
    
    // if not hit, return background (looking up the texture by converting the ray direction to latlong around y)
    if(not intersection.hit) {
        return eval_env(scene->background, scene->background_txt, ray.d);
    }
    
    // setup variables for shorter code
    auto pos = intersection.pos;
    auto norm = intersection.norm;
    auto v = -ray.d;
    
    // compute material values by looking up textures
    auto ke = lookup_scaled_texture(intersection.mat->ke, intersection.mat->ke_txt, intersection.texcoord, true);
    auto kd = lookup_scaled_texture(intersection.mat->kd, intersection.mat->kd_txt, intersection.texcoord, true);
    auto ks = lookup_scaled_texture(intersection.mat->ks, intersection.mat->ks_txt, intersection.texcoord, true);
    auto n = intersection.mat->n;
    auto mf = intersection.mat->microfacet;
    
    // accumulate color starting with ambient
    auto c = scene->ambient * kd;
    
    // add emission if on the first bounce
    if(depth == 0 and dot(v,norm) > 0) c += ke;
    
    // foreach point light
    for(auto light : scene->lights) {
        // compute light response
        auto cl = light->intensity / (lengthSqr(light->frame.o - pos));
        // compute light direction
        auto l = normalize(light->frame.o - pos);
        // compute the material response (brdf*cos)
        auto brdfcos = max(dot(norm,l),0.0f) * eval_brdf(kd, ks, n, v, l, norm, mf);
        // multiply brdf and light
        auto shade = cl * brdfcos;
        // check for shadows and accumulate if needed
        if(shade == zero3f) continue;
        // if shadows are enabled
        if(scene->path_shadows) {
            // perform a shadow check and accumulate
            if(not intersect_shadow(scene, ray3f::make_segment(pos,light->frame.o))) c += shade;
        } else {
            // else just accumulate
            c += shade;
        }
    }

    // foreach surface
    for(Surface * s: scene->surfaces){

        // skip if no emission from surface
        if(s->mat->ke==zero3f){
            continue;
        }

        // todo: pick a point on the surface, grabbing normal, area, and texcoord
        // generate a 2d random number
        vec2f r_num = rng->next_vec2f();

        // compute light position, normal, area
        vec3f light_pos;

        vec3f light_norm;

        float light_area;

        // check if quad
        if(s->isquad){
           light_pos = s->frame.o + (r_num.x-0.5f)*2.0f*s->radius*s->frame.x + (r_num.y-0.5f)*2.0f*s->radius*s->frame.y;

           light_norm = normalize(s->frame.z);

           light_area = 4.0f*pow(s->radius, 2);
        }
        else {  // else if sphere

            vec3f dir = normalize(sample_direction_spherical_uniform(r_num));

            light_pos = s->frame.o + s->radius*dir;

            light_norm = normalize(dir);

            light_area = 4.0f*pif*pow(s->radius, 2);
        }

        // set tex coords as random value got before
        intersection.texcoord=r_num;

        // get light emission from material and texture
        vec3f kes = lookup_scaled_texture(s->mat->ke, s->mat->ke_txt, intersection.texcoord, true);

         // compute light direction
        vec3f ldir = normalize(light_pos - pos);

        // compute light response (ke * area * cos_of_light / dist^2)
        vec3f light_response = kes*light_area*max(dot(light_norm,-ldir),0.0f)/lengthSqr(light_pos-pos);

        // compute the material response (brdf*cos)
        vec3f material_repsonse = max(dot(norm,ldir),0.0f)*eval_brdf(kd, ks, n, v, ldir, norm, mf);

        // multiply brdf and light
        vec3f res = light_response*material_repsonse;

        // check for shadows and accumulate if needed
        // if shadows are enabled
        if(scene->path_shadows){
            // perform a shadow check and accumulate
            if(!intersect_shadow(scene, ray3f::make_segment(pos, light_pos))){
                c+=res;
            }
        } else{
        // else just accumulate
            c+=res;
        }
    }
    
    // todo: sample the brdf for environment illumination if the environment is there
    // if scene->background is not zero3f
    if(scene->background!=zero3f){
        // pick direction and pdf
        pair<vec3f, float> samp_brdf = sample_brdf(kd, ks, n, v, norm, rng->next_vec2f(), rng->next_float());
        vec3f dir = samp_brdf.first;
        float pdf = samp_brdf.second;

        // compute the material response (brdf*cos)
        vec3f env = eval_env(scene->background, scene->background_txt, dir);
        //printf("%f %f %f\n", env.x, env.y, env.z);
        //vec3f material_repsonse = max(dot(norm, dir),0.0f)*eval_env(intersection.mat->ke,intersection.mat->ke_txt, dir);
        vec3f material_repsonse = max(dot(norm, dir), 0.0f)*(eval_brdf(kd, ks, n, v, dir, norm, mf));

        // todo: accumulate response scaled by brdf*cos/pdf
         vec3f res = env*material_repsonse/pdf;

        // if material response not zero3f
         if(material_repsonse!=zero3f){

             // if shadows are enabled
             if(scene->path_shadows){
                 // perform a shadow check and accumulate
                 if(!intersect_shadow(scene, ray3f(pos, dir))){
                     c+=res;
                 }
             } else{
             // else just accumulate
                 c+=res;
             }
         }
    }
    
    // todo: sample the brdf for indirect illumination
    // if kd and ks are not zero3f and haven't reach max_depth
    if((kd!=zero3f || ks!=zero3f) && (depth < scene->path_max_depth)){
        // pick direction and pdf
        pair<vec3f, float> samp_brdf = sample_brdf(kd, ks, n, v, norm, rng->next_vec2f(), rng->next_float());
        vec3f dir = samp_brdf.first;
        float pdf = samp_brdf.second;
        // compute the material response (brdf*cos)
         vec3f material_repsonse = max(dot(norm, dir),0.0f)*eval_brdf(kd, ks, n, v, dir, norm, mf);
         vec3f res = material_repsonse/pdf;
        // accumulate recersively scaled by brdf*cos/pdf
         if(not(intersection.mat->ks == zero3f)){
             c = pathtrace_ray(scene, ray3f(pos, dir), rng, depth+1)*res;
         }
        c+= pathtrace_ray(scene, ray3f(pos, dir), rng, depth+1)*res;
    }

    // if the material has reflections
    if(not (intersection.mat->kr == zero3f)) {
        // create the reflection ray
        auto rr = ray3f(intersection.pos,reflect(ray.d,intersection.norm));
        if(blur){
           //auto random = rng->next_vec2f();

           int N = 3; //samples
           auto rri = rr;
           rri.d += blurSize * (vec3f(.5,.5,.5) - rng->next_vec3f());
           //auto rri = rr+(.5-random.x)*blurSize*u+(.5-random.y)*blurSize*v;
           //rri = normalize(rri);
           auto blurSum = vec3f();
           for(int i=0; i<N; i++){
               blurSum += pathtrace_ray(scene, rri, rng, depth+1);
           }
           c += (intersection.mat->kr/N) * blurSum;
       }
       else{
            // accumulate the reflected light (recursive call) scaled by the material reflection
            c += intersection.mat->kr * pathtrace_ray(scene,rr,rng,depth+1);
        }
    }
    
    // return the accumulated color
    return c;
}


// runs the raytrace over all tests and saves the corresponding images
int main(int argc, char** argv) {
    auto args = parse_cmdline(argc, argv,
        { "04_pathtrace", "raytrace a scene",
            {  {"resolution",     "r", "image resolution", typeid(int),    true,  jsonvalue() } },
            {  {"scene_filename", "",  "scene filename",   typeid(string), false, jsonvalue("scene.json") },
               {"image_filename", "",  "image filename",   typeid(string), true,  jsonvalue("") } }
        });
    
    auto scene_filename = args.object_element("scene_filename").as_string();
    Scene* scene = nullptr;
    if(scene_filename.length() > 9 and scene_filename.substr(0,9) == "testscene") {
        int scene_type = atoi(scene_filename.substr(9).c_str());
        scene = create_test_scene(scene_type);
        scene_filename = scene_filename + ".json";
    } else {
        scene = load_json_scene(scene_filename);
    }
    error_if_not(scene, "scene is nullptr");
    
    auto image_filename = (args.object_element("image_filename").as_string() != "") ?
        args.object_element("image_filename").as_string() :
        scene_filename.substr(0,scene_filename.size()-5)+".png";
    
    if(not args.object_element("resolution").is_null()) {
        scene->image_height = args.object_element("resolution").as_int();
        scene->image_width = scene->camera->width * scene->image_height / scene->camera->height;
    }
    
    // NOTE: acceleration structure does not support animations
    message("reseting animation...\n");
    animate_reset(scene);

//    for (Mesh* m: scene->meshes) {
//            vector<vec3f> poss = vector<vec3f>(m->pos.size(), zero3f);
//            int i = 0;
//            for (vec3f pos : m->pos) {
//                poss[i] = pos * 10;
//                i++;
//            }
//            m->pos = poss;

//            facet_normals(m);
//        }
    
    message("accelerating...\n");
    accelerate(scene);
    
    message("rendering %s...\n", scene_filename.c_str());
    auto image = pathtrace(scene, parallel_pathtrace);
    
    message("saving %s...\n", image_filename.c_str());
    write_png(image_filename, image, true);
    
    delete scene;
    message("done\n");
}


/////////////////////////////////////////////////////////////////////
// Rendering Code


// pathtrace an image
void pathtrace(Scene* scene, image3f* image, RngImage* rngs, int offset_row, int skip_row, bool verbose) {
    if(verbose) message("\n  rendering started        ");
    // foreach pixel
    for(auto j = offset_row; j < scene->image_height; j += skip_row ) {
        if(verbose) message("\r  rendering %03d/%03d        ", j, scene->image_height);
        for(auto i = 0; i < scene->image_width; i ++) {
            // init accumulated color
            image->at(i,j) = zero3f;
            // grab proper random number generator
            auto rng = &rngs->at(i, j);
            // foreach sample
            for(auto jj : range(scene->image_samples)) {
                for(auto ii : range(scene->image_samples)) {
                    // compute ray-camera parameters (u,v) for the pixel and the sample
                    auto u = (i + (ii + rng->next_float())/scene->image_samples) /
                        scene->image_width;
                    auto v = (j + (jj + rng->next_float())/scene->image_samples) /
                        scene->image_height;
                    // compute camera ray
                    auto ray = transform_ray(scene->camera->frame,
                        ray3f(zero3f,normalize(vec3f((u-0.5f)*scene->camera->width,
                                                     (v-0.5f)*scene->camera->height,-1))));
                    // set pixel to the color raytraced with the ray
                    image->at(i,j) += pathtrace_ray(scene,ray,rng,0);
                }
            }
            // scale by the number of samples
            image->at(i,j) /= (scene->image_samples*scene->image_samples);
            //Correct of Russian Roulette light loss
            image->at(i,j) *= 1/(1-russian_threshold);
        }
    }
    if(verbose) message("\r  rendering done        \n");
    
}

// pathtrace an image with multithreading if necessary
image3f pathtrace(Scene* scene, bool multithread) {
    // allocate an image of the proper size
    auto image = image3f(scene->image_width, scene->image_height);
    
    // create a random number generator for each pixel
    auto rngs = RngImage(scene->image_width, scene->image_height);

    // if multitreaded
    if(multithread) {
        // get pointers
        auto image_ptr = &image;
        auto rngs_ptr = &rngs;
        // allocate threads and pathtrace in blocks
        auto threads = vector<thread>();
        auto nthreads = thread::hardware_concurrency();
        for(auto tid : range(nthreads)) threads.push_back(thread([=](){
            return pathtrace(scene,image_ptr,rngs_ptr,tid,nthreads,tid==0);}));
        for(auto& thread : threads) thread.join();
    } else {
        // pathtrace all rows
        pathtrace(scene, &image, &rngs, 0, 1, true);
    }
    
    // done
    return image;
}


