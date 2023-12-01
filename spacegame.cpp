//************************************************************************* //
// Space Shooting Game
//
// Szirmay-Kalos L�szl�
//************************************************************************* //
#include "gameengine.h"

const float fNewton = 0.003f;     
const float rocketPower = 0.03f;
int score = 0;

//--------------------------------------------
class Space : public GameObject { 
//--------------------------------------------
	Shader   * shader;
	Geometry * geometry;
	Texture * texture;
public:
	Space( Shader * _shader ) { 
		shader = _shader;
		texture = new Texture("stars.bmp");
		geometry = new BoxSurface(100);
	}
	Shader * getShader() { return shader; }
	Texture * getTexture() { return texture; }
	Geometry * getGeometry() { return geometry; }

};

//--------------------------------------------
class Planet : public GameObject { 
//--------------------------------------------
	Texture * texture;
	Material * material;
protected:
    float   mass, dist, axis_angle;
    float   rot_angle, rot_speed, rev_angle, rev_speed; 
 public:
	static Shader   * shader;
	static Geometry * geometry;

    Planet(Texture * _texture, Material * _material, vec3 p0, float R, float axis_angle0 = 0) {
		texture = _texture;
		material = _material;
        boundingRadius = R;      
        mass = pow(R, 3) * 20;    
        axis_angle = axis_angle0 * (float)M_PI/180.0f; 
        rot_angle = rev_angle = 0;
		rot_speed = 1;        

		position = p0;
		dist = length(position);
		if (dist > 0.1) {
			const float sunAccelRef = 10;
			float sun_accel = sunAccelRef / pow(dist, 2);
			rev_speed = sqrt(sun_accel / dist);
		}
		else rev_speed = 0;
		rev_angle = Rand((float)M_PI, (float)M_PI);
    }
	Shader * getShader() { return shader; }
	Texture * getTexture() { return texture; }
	Geometry * getGeometry() { return geometry; }
	Material * getMaterial() { 
		return material; 
	}

	float Radius( ) { return boundingRadius; }
    float Mass( ) { return mass; }

	void Animate(float tstart, float tend) {
		float dt = tend - tstart;
		rot_angle += rot_speed * dt;   
		if (rot_angle > 2 * (float)M_PI) rot_angle -= 2 * (float)M_PI;
		rev_angle += rev_speed * dt;   
		if (rev_angle > 2 * (float)M_PI) rev_angle -= 2 * (float)M_PI;
		position.x = dist * cos(rev_angle); position.y = dist * sin(rev_angle); position.z = 0;
    }
	void SetModelingTransformation(RenderState& state) {
		state.M = ScaleMatrix(vec3(boundingRadius, boundingRadius, boundingRadius)) *
			      RotationMatrix(rot_angle, vec3(0, 0, 1)) * RotationMatrix(axis_angle, vec3(1, 0, 0)) * 
			      TranslateMatrix(position);
		state.Minv = TranslateMatrix(-position) *
				  RotationMatrix(-axis_angle, vec3(1, 0, 0)) * RotationMatrix(-rot_angle, vec3(0, 0, 1)) *
				  ScaleMatrix(vec3(1 / boundingRadius, 1 / boundingRadius, 1 / boundingRadius));
	}
};

//--------------------------------------------
class ExplosionParticle : public Particle { 
//--------------------------------------------
public:
	ExplosionParticle() {
		position = vec3(0, 0, 0);                
		time_to_live = Rand(1.5f, 0.8f);      
		size = 0.01f;                   
		dsize = Rand(0.3f, 0.2f) / time_to_live; 
		do { velocity = vec3(Rand(0, 1), Rand(0, 1), Rand(0, 1)); } while (length(velocity) > 1);
		velocity = velocity * 1.5;
		do { acceleration = vec3(Rand(0, 1), Rand(0, 1), Rand(0, 1)); } while (length(acceleration) > 1);
		acceleration = acceleration * 0.5;
		color = vec4(Rand(1, 0), Rand(0.8f, 0.4f), Rand(0.3f, 0.2f), 0.5f);
		dcolor = vec4(0, -0.25f, -0.3f, -1) / time_to_live;
	}
};

//--------------------------------------------
class Explosion : public ParticleSystem { 
//--------------------------------------------
public:
	Explosion(vec3 pos0) : ParticleSystem(pos0) { 
		for (int i = 0; i < 500; i++) AddParticle(new ExplosionParticle()); 
	}
};

//--------------------------------------------
class Ship : public GameObject { 
//--------------------------------------------
	float mass;                       
	vec3  gravity_force, rocket_force; 
	float last_shot, closest_planet_dist;          
	vec3 up, left;

	enum AI_State {            
		ESCAPE_FROM_PLANET,     
		ESCAPE_FROM_AVATAR,     
		CHASE_AVATAR            
	} ai_state;

public:
	static Shader *   shader;
	static Material * material;
	static Texture *  texture;
	static Geometry * geometry;

	Ship(vec3& pos0) {
		boundingRadius = 0.3f;      
		mass = 0.1f;
		position = pos0;
		velocity = vec3(0.1f, 0.2f, 0.1f);
		up = vec3(0, 1, 0);
		ai_state = CHASE_AVATAR;
		last_shot = 0;
	}
	Shader * getShader() { return shader; }
	Texture * getTexture() { return texture; }
	Geometry * getGeometry() { return geometry; }
	Material * getMaterial() { return material; }


	void Control(float tstart, float tend) {
		float dt = tend - tstart;
		float v = length(velocity);
		if (v > 1) velocity = velocity * (1 / v);
		closest_planet_dist = 20;            
		gravity_force = vec3(0, 0, 0);       
		scene->Interact(this);                
		acceleration = (gravity_force + rocket_force) * (1 / mass);
		last_shot += dt;
		left = cross(up * 0.99f + normalize(acceleration) * 0.01f, velocity);
		up = cross(velocity, left);
	}

	void Interact(GameObject * object);

	void SetModelingTransformation(RenderState& state) {
		vec3 head = normalize(velocity);
		up = normalize(up);
		left = cross(up, head);
		state.M = mat4(left.x, left.y, left.z, 0,
			          up.x, up.y, up.z, 0,
			          head.x, head.y, head.z, 0,
			          0, 0, 0, 1) * TranslateMatrix(position);
		state.Minv = TranslateMatrix(-position) *
			        mat4(left.x, up.x, head.x, 0,
				         left.y, up.y, head.y, 0,
				         left.z, up.z, head.z, 0,
				         0, 0, 0, 1);
	}

	void Kill() {
		printf("\n Ship is destroyed ");
		score++;
		GameObject::Kill();
	}
};

//--------------------------------------------
class PhotonRocket : public BillBoard { 
//--------------------------------------------
	float   age;           
public:
	static Texture * texture;

	PhotonRocket(vec3& pos0, vec3& shooter_velocity) : BillBoard(pos0, 0.3f) {
		velocity = shooter_velocity + normalize(shooter_velocity) * 2;
		age = 0;
	}
	Texture * getTexture() { return texture; }


	void Control(float tstart, float tend) {    
		float dt = tend - tstart;
		float hit_time = dt;  
		vec3 hit_point;     
		GameObject * hit_object = scene->Collide(this, hit_time, hit_point);

		if (hit_object) {								
			Kill();										
			if (dynamic_cast<Planet*>(hit_object) || dynamic_cast<Ship*>(hit_object)) {
 				scene->Join(new Explosion(hit_point));	
			}
			if (dynamic_cast<Avatar*>(hit_object) || dynamic_cast<Ship*>(hit_object)) {
				hit_object->Kill();						
			}
		}
		age += dt;                 
		if (age > 10) Kill();      
	}
	void Animate(float tstart, float tend) { GameObject::Animate(tstart, tend); }
};

//--------------------------------------------
 class Self : public Avatar { 
//--------------------------------------------
    int     lifes;                       
    vec3	gravity_force, rocket_force; 
	float   mass, last_shot, last_turn;
 public:
    Self(vec3& pos0) : Avatar(pos0) { 
        velocity = vec3(-1, 0, 0);     
        mass = 0.1f;                     
        lifes = 5;                       
        boundingRadius = 0.5f;         
		last_shot = last_turn = 0;
    }

	void Control(float tstart, float tend) { 
		float dt = tend - tstart;
		if (!alive) return;
		vec3 left = cross(wVup * 0.99f + acceleration * 0.01f, velocity);
		wVup = cross(velocity, normalize(left));

        float v = length(velocity);
        if (v > 1) velocity = velocity * (1/v);
        gravity_force = vec3(0, 0, 0);
        scene->Interact(this);
        acceleration = (gravity_force + rocket_force) * (1 / mass);
		last_shot += dt;
		last_turn += dt;
    }

    void Interact( GameObject * object ) {
        if ( dynamic_cast<Planet *>(object) ) {        // a bolyg�k vonzz�k az urhaj�t
            Planet * planet = (Planet *)object;
            vec3 dr = planet->position - position; // relat�v helyzet
            float dist = length(dr);                // t�vols�g
            gravity_force = gravity_force + dr * (fNewton * mass * planet->Mass() / pow(dist, 3));
            if (dist < planet->Radius() ) Kill();  // a bolyg�val �tk�z�s fat�lis
        }   
    }
 
    void ProcessInput( ) {
        if ( alive ) {
			vec3 w = normalize(-velocity);
			vec3 u = normalize(cross(wVup, w));
			vec3 v = cross(w, u);

			rocket_force = w * -rocketPower * 2;
			if (input.GetKeyStatus(GLUT_KEY_UP))    rocket_force = rocket_force - v * rocketPower;
			if (input.GetKeyStatus(GLUT_KEY_DOWN))  rocket_force = rocket_force + v * rocketPower;
			if (input.GetKeyStatus(GLUT_KEY_LEFT))  rocket_force = rocket_force - u * rocketPower;
			if (input.GetKeyStatus(GLUT_KEY_RIGHT)) rocket_force = rocket_force + u * rocketPower;

			if (input.GetKeyStatus('w') && last_turn > 0.5) {
				velocity = -velocity;
				last_turn = 0;
			}
			if (input.GetKeyStatus(' ') && last_shot > 1) { 
                scene->Join(new PhotonRocket(position + normalize(velocity) * boundingRadius * 2, velocity));
				last_shot = 0;
			}
        } else {
            if ( input.GetKeyStatus('s') ) {
                if ( lifes > 0 ) {
                    position = vec3(0, 9, 3);
                    velocity = vec3(0, -1, 0);
                    alive = true;
                }
            }
        }
    }

    void Kill() {
        if ( alive ) {
			printf("Lifes: %d\n", --lifes);
//            alive = false;
        }
    }
 };

Shader * Ship::shader = NULL;
Material * Ship::material = NULL;
Texture * Ship::texture = NULL;
Geometry * Ship::geometry = NULL;

Shader * Planet::shader = NULL;
Geometry * Planet::geometry = NULL;

Texture * PhotonRocket::texture = NULL;

void Ship::Interact(GameObject * object) {
	if (dynamic_cast<Planet*>(object)) {  
		Planet * planet = (Planet *)object;
		vec3 dr = planet->position - position; 
		float dist = length(dr);               
		dr = normalize(dr);
		gravity_force = gravity_force + dr * (fNewton * mass * planet->Mass() / pow(dist, 3));

		if (dist < planet->Radius()) {           
			Kill();                              
			scene->Join(new Explosion(position)); 
		}
		if (dist < closest_planet_dist) {      
			closest_planet_dist = dist;
			if (dist < planet->Radius() * 3) ai_state = ESCAPE_FROM_PLANET;
			rocket_force = dr * (-rocketPower); 
		}
	}
	if (closest_planet_dist > 4) ai_state = CHASE_AVATAR;

	if (dynamic_cast<Avatar*>(object)) {     
		Avatar * avatar = (Avatar *)object;
		vec3 dr = avatar->position - position; 
		float dist = length(dr);               
		dr = normalize(dr);                   
		vec3 head = normalize(velocity);       
		vec3 avatar_head = normalize(avatar->velocity); 

		switch (ai_state) {                   
		case ESCAPE_FROM_AVATAR:
			if (-dot(dr, avatar_head) < 0.5 && dist > 4) ai_state = CHASE_AVATAR;
			else rocket_force = cross(avatar_head, dr) * rocketPower;
			break;
		case CHASE_AVATAR:
			if (-dot(dr, avatar_head) > 0.5 || dist < 4) ai_state = ESCAPE_FROM_AVATAR;
			rocket_force = dr * rocketPower;
			break;
		}

		if (last_shot > 2 && dot(head, dr) > 0.9f && dist < 8) {
			vec3 start = position + normalize(velocity) * boundingRadius * 2.0f;
			scene->Join(new PhotonRocket(start, velocity));
			last_shot = 0;
		}
	}
}

//--------------------------------------------
void Scene :: Build() { 
//--------------------------------------------
	Shader * surfaceShader = new SurfaceShader();
	Shader * volumetricShader = new VolumetricShader();

	avatar = new Self(vec3(20, 0, 1));
	Join(avatar);		// avatar
	Join(new Space(surfaceShader));	// space

	Material * shinyMaterial = new Material();
	shinyMaterial->ka = vec3(0.3f, 0.3f, 0.3f);
	shinyMaterial->kd = vec3(1, 1, 1);
	shinyMaterial->ks = vec3(3, 3, 3);
	shinyMaterial->shininess = 200;

	Ship::shader = surfaceShader;	
	Ship::material = shinyMaterial;
	Ship::texture = new Texture("ship_texture.bmp");
	Ship::geometry = new OBJSurface("ship.obj", 0.1f);

	Join(new Ship(vec3(5, 3, 10)));	 // ships
	Join(new Ship(vec3(-5, 3, 0)));
	Join(new Ship(vec3(5, -3, 0)));
	Join(new Ship(vec3(-5, 3, -10)));

	Material * diffuseMaterial = new Material();
	diffuseMaterial->ka = vec3(0.1f, 0.1f, 0.1f);
	diffuseMaterial->kd = vec3(1, 1, 1);

	Planet::shader = surfaceShader;
	Planet::geometry = new Sphere();

	Light light;
	light.La = vec3(1.5, 1.5, 1.5); light.Le = vec3(2, 2, 2); light.wLightPos = vec4(0, 0, 0, 1);
	lights.push_back(light);

	Material * sunMaterial = new Material();
	sunMaterial->emission = 1;

	Planet * sun = new Planet(new Texture("sun.bmp"), sunMaterial, vec3(0, 0, 0), 3.0f);			// Sun
	Join(sun);			// Sun

	Join(new Planet(new Texture("mercury.bmp"), diffuseMaterial, vec3(6, 0, 0), 0.2f));		// Mercury
	Join(new Planet(new Texture("venus.bmp"), diffuseMaterial, vec3(7, 0, 0), 0.4f));		// Venus
	Join(new Planet(new Texture("earth.bmp"), diffuseMaterial, vec3(8, 0, 0), 0.5f, 23));    // Earth
	Join(new Planet(new Texture("mars.bmp"), diffuseMaterial, vec3(10, 0, 0), 0.3f, 25));
	Join(new Planet(new Texture("jupiter.bmp"), diffuseMaterial, vec3(11, 0, 0), 0.7f, 3));
	Join(new Planet(new Texture("saturn.bmp"), diffuseMaterial, vec3(12, 0, 0), 0.6f, 26));
	Join(new Planet(new Texture("uranus.bmp"), diffuseMaterial, vec3(13, 0, 0), 0.5f, 82));
	Join(new Planet(new Texture("neptun.bmp"), diffuseMaterial, vec3(15, 0, 0), 0.5f, 29));
	Join(new Planet(new Texture("pluto.bmp"), diffuseMaterial, vec3(17, 0, 0), 0.2f, 62));

	PhotonRocket::texture = new Texture("photon.bmp", true);
};
